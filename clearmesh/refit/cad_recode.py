"""CAD-Recode wrapper — point cloud -> CadQuery code -> refit mesh.

Upstream: https://github.com/filaPro/cad-recode  (ICCV 2025)
Weights:  https://huggingface.co/filapro/cad-recode-v1.5  (CC-BY-NC-4.0)
Paper:    https://arxiv.org/abs/2412.14042

Licence note: the weights are CC-BY-NC-4.0 (non-commercial). Any use of
this module in a shipped product requires either retraining from scratch
on a commercial-compatible dataset or direct permission from the authors.

Architecture: CAD-Recode is a Qwen2-1.5B language model with a small
Fourier point-encoder + linear projector grafted onto the input
embedding stream. Point positions are projected into the LM's hidden
space and spliced into the token sequence at positions where
``attention_mask == -1``. The LM then autoregressively generates a
CadQuery Python script that, when ``exec()``'d, builds a ``cq.Workplane``
solid stored in a global named ``r``.

Input spec (non-negotiable):
  - shape        (1, 256, 3)
  - dtype        float32
  - frame        centred at origin, scaled so ``max(extents) == 2``
  - sampling     FPS from 8192 surface samples (matches training)
  - normals      NOT used

Output spec:
  - CadQuery script, 50-300 tokens typical, 768 max
  - Convention: script assigns final solid to variable ``r``
  - Invalidity (code that fails to exec): <1% on paper benchmarks

Quality expectations (from paper Table 1 / Fig 5):
  - Mechanical prismatic meshes (boxes, cylinders, holes): IoU 70-92%
  - Organic meshes (characters, plants, creatures): DEGRADES severely —
    training data is 1M procedural prismatic shapes; no signal for
    smooth curvature or non-axis-aligned freeform surfaces.
  - Expect "stack of boxes approximating the bounding box" for organic
    input. This is a known failure mode, not a bug.

Typical latency on H100:
  - 1 candidate: 1-3s (flash-attn)
  - 10 candidates (paper default for scoring): 10-30s

Usage:
    from clearmesh.refit import CadRecodeRefiner

    refiner = CadRecodeRefiner()  # lazy-loads weights on first .refit() call
    result = refiner.refit(my_mesh, n_candidates=1)
    print(result.cadquery_code)    # editable Python script
    result.refit_mesh.export("refit.glb")
"""

from __future__ import annotations

import importlib
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh


__all__ = ["CadRecodeRefiner", "CadRecodeResult"]


@dataclass
class CadRecodeResult:
    """Output of a CadRecodeRefiner.refit() call."""

    # Primary artefact — editable, diffable, interpretable CadQuery script.
    cadquery_code: str

    # The solid after tessellation (may be None if exec failed and no
    # fallback succeeded; in that case ``cadquery_code`` is still set so
    # callers can retry or show the script to the user).
    refit_mesh: Optional[trimesh.Trimesh]

    # The 256-point cloud actually fed to the model (post-FPS,
    # post-normalisation). Useful for visualisation / debugging.
    point_cloud: np.ndarray

    # Transform applied during normalisation; callers can use this to map
    # the refit mesh back into the original frame.
    #   original_pt = refit_pt * inv_scale + input_centroid
    input_centroid: np.ndarray
    input_scale: float  # max(extents); divide input by this to normalise

    # Symmetric Chamfer to input point cloud (only populated when
    # n_candidates > 1 and scoring was performed).
    chamfer: Optional[float] = None

    # Other sampled candidates when n_candidates > 1. Ordered by chamfer
    # (best first). Each entry: (cadquery_code, refit_mesh, chamfer).
    candidates: list = field(default_factory=list)


# --- Fourier point encoder + CADRecode LM class (adapted from demo.ipynb) ---
# These live inline here so we don't need a working clone of filaPro/cad-recode
# on the machine. Matches the paper's §4.3: 8 frequency bands, 51-dim input,
# linear to hidden_size=1536.

def _build_cad_recode_classes():
    """Imports torch and transformers lazily and constructs the custom
    CADRecode LM class. Returns (FourierPointEncoder, CADRecode).

    v2 strategy (simpler, avoids transformers-5.x-specific issues):
      - Subclass Qwen2ForCausalLM minimally — just add point_encoder.
      - DO NOT override forward(). Do all point-splicing OUTSIDE the
        model by precomputing inputs_embeds and passing them in
        directly. This sidesteps the attention_mask==-1 convention
        entirely and lets us use the vanilla generate() path without
        fighting KV-cache/position-id handling.
      - prepare_inputs_for_generation is still overridden to thread
        inputs_embeds through the first forward only (subsequent
        autoregressive steps use input_ids like any other LM).
    """
    import torch
    from torch import nn
    from transformers import Qwen2ForCausalLM, Qwen2Model, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast

    class FourierPointEncoder(nn.Module):
        """Projects (B, N, 3) xyz into (B, N, hidden_size).

        Input:   xyz coordinates in [-1, 1] (normalised cube-of-2)
        Features: raw xyz ++ sin(2^k * xyz) ++ cos(2^k * xyz)  for k in 0..7
                  => 3 + 3*8 + 3*8 = 51 dims
        Output:  Linear(51, hidden_size) projection
        """

        def __init__(self, hidden_size: int):
            super().__init__()
            frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
            self.register_buffer("frequencies", frequencies, persistent=False)
            self.projection = nn.Linear(51, hidden_size)

        def forward(self, points: "torch.Tensor") -> "torch.Tensor":
            x = (points.unsqueeze(-1) * self.frequencies).view(
                *points.shape[:-1], -1
            )
            x = torch.cat((points, x.sin(), x.cos()), dim=-1)
            return self.projection(x)

    class CADRecode(Qwen2ForCausalLM):
        """Qwen2-1.5B with Fourier point splicing.

        Positions in the input where ``attention_mask == -1`` are replaced
        with projected point embeddings. The -1s get flipped to 1 in-place
        so subsequent generate() steps work on a clean mask.

        NOTE on init: the upstream demo.ipynb calls
        ``PreTrainedModel.__init__(self, config)`` then manually creates
        ``self.model`` and ``self.lm_head``. That worked on transformers
        4.45, but on transformers 5.x this detaches lm_head from the
        embed_tokens tie, so the checkpoint's missing lm_head.weight
        stays random and the model emits garbage. We just call the
        parent's ``__init__`` (which gives us proper tie_word_embeddings
        behavior) and add the point_encoder on top.
        """

        def __init__(self, config):
            super().__init__(config)
            self.point_encoder = FourierPointEncoder(config.hidden_size)

        def encode_point_cloud(
            self,
            input_ids: "torch.Tensor",       # (1, N+1) int
            point_cloud: "torch.Tensor",     # (1, N, 3) float
            text_start_token_id: int,        # typically <|im_start|>
        ) -> "torch.Tensor":
            """Build the (1, N+1, hidden) embedding tensor used as the
            first-call ``inputs_embeds`` for generate().

            The convention:
              - First N positions = projected point embeddings.
              - Position N+1 = the <|im_start|> token's vanilla embedding.
            """
            embed = self.model.embed_tokens
            pc_f32 = point_cloud.to(torch.float32)
            point_embeds = self.point_encoder(pc_f32).to(embed.weight.dtype)
            start_embed = embed(
                torch.tensor([[text_start_token_id]], device=input_ids.device)
            )
            return torch.cat([point_embeds, start_embed], dim=1)

    return FourierPointEncoder, CADRecode


class CadRecodeRefiner:
    """Wrap filapro/cad-recode-v1.5 for ClearMesh.

    The model is **hardcoded to 256 input points** — the paper's ablation
    tested 128 vs 256 and found no benefit beyond 256. We silently
    downsample larger point clouds.

    Licence: model weights are CC-BY-NC-4.0. RESEARCH ONLY.
    """

    N_POINTS = 256
    N_PRE_POINTS = 8192
    CUBE_SIDE = 2.0

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "filapro/cad-recode-v1.5",
        tokenizer_id: str = "Qwen/Qwen2-1.5B",
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "auto",
    ):
        self.device = device
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.attn_implementation = attn_implementation
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    def _load(self):
        """Lazy-load the LM + tokenizer on first use."""
        import torch
        from transformers import AutoTokenizer

        _, CADRecode = _build_cad_recode_classes()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            pad_token="<|im_end|>",
            padding_side="left",
        )

        # Fall back to sdpa if flash-attn isn't installed
        try:
            self._model = CADRecode.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.attn_implementation,
            )
        except (ImportError, ValueError) as e:
            warnings.warn(
                f"[cad_recode] falling back to attn='sdpa' ({e})"
            )
            self._model = CADRecode.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                attn_implementation="sdpa",
            )

        # Force point_encoder to float32 — its internal math mixes a
        # float32 buffer (self.frequencies) with whatever dtype
        # from_pretrained cast the Linear to, and the bf16-projection /
        # float32-buffer combination triggers F.linear dtype errors.
        self._model.point_encoder.float()

        # CRITICAL: the v1.5 checkpoint does not include lm_head.weight
        # (it's tied to embed_tokens). Transformers 5.x doesn't always
        # re-tie after from_pretrained for subclassed causal LMs.
        # Explicitly force tying so the LM head uses the trained
        # embeddings instead of the randomly initialized Linear.
        try:
            self._model.tie_weights()
        except Exception:
            # Fallback: manually wire lm_head.weight = embed_tokens.weight
            self._model.lm_head.weight = self._model.model.embed_tokens.weight

        self._model = self._model.to(self.device).eval()

    # --- Point cloud prep -------------------------------------------------

    def _normalise(self, mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray, float]:
        """Centre at origin, scale so max extent = 2.

        Returns (normalised_mesh, input_centroid, input_scale). Caller can
        map refit_mesh back by:
            refit.vertices * input_scale + input_centroid
        """
        centroid = mesh.bounding_box.centroid.copy()
        m = mesh.copy()
        m.apply_translation(-centroid)
        ext = max(m.extents)
        scale = ext if ext > 0 else 1.0
        m.apply_scale(self.CUBE_SIDE / scale)
        return m, np.asarray(centroid), float(scale / self.CUBE_SIDE)

    @staticmethod
    def _fps_torch(points: "torch.Tensor", k: int) -> "torch.Tensor":
        """Greedy farthest-point-sampling in pure torch (no pytorch3d).

        points: (N, 3) tensor. Returns k selected indices.

        This replaces ``pytorch3d.ops.sample_farthest_points`` so we
        don't need to compile pytorch3d against torch 2.6+cu124 (for
        which no prebuilt wheels exist). The algorithm is the textbook
        greedy FPS: start from point 0, repeatedly pick the point
        farthest from the current set.
        """
        import torch

        n = points.shape[0]
        device = points.device
        selected = torch.zeros(k, dtype=torch.long, device=device)
        # distance from each point to the selected set
        dist = torch.full((n,), float("inf"), device=device)
        # start from index 0 (deterministic) — matches pytorch3d default
        cur = 0
        for i in range(k):
            selected[i] = cur
            d = ((points - points[cur]) ** 2).sum(dim=-1)
            dist = torch.minimum(dist, d)
            cur = int(torch.argmax(dist).item())
        return selected

    def _sample_points(
        self,
        mesh: trimesh.Trimesh,
        n_points: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Surface-sample + FPS to n_points (no pytorch3d dependency)."""
        import torch

        if seed is not None:
            # trimesh.sample uses numpy global RNG
            np_state = np.random.get_state()
            np.random.seed(seed)
            try:
                verts, _ = trimesh.sample.sample_surface(mesh, self.N_PRE_POINTS)
            finally:
                np.random.set_state(np_state)
        else:
            verts, _ = trimesh.sample.sample_surface(mesh, self.N_PRE_POINTS)

        v_t = torch.as_tensor(verts, dtype=torch.float32)
        # Use GPU for FPS if available — ~5x faster on 8192x256 scale
        if torch.cuda.is_available():
            v_t = v_t.cuda()
        ids = self._fps_torch(v_t, n_points).cpu().numpy()
        picked = verts[ids]
        return np.asarray(picked, dtype=np.float32)

    # --- Inference --------------------------------------------------------

    def _generate_one(
        self,
        point_cloud: np.ndarray,
        max_new_tokens: int,
    ) -> str:
        """Run one model.generate() pass.

        v2 path: we build the spliced inputs_embeds BEFORE calling
        generate(), pass those directly, and let the LM run as a
        vanilla Qwen2 from there. This avoids the -1-mask convention
        and the custom forward() that caused looping / empty output
        on transformers 5.x.

        The resulting input to generate() is:
          inputs_embeds : (1, N+1, hidden)  — N point embeddings + <|im_start|>
          attention_mask: (1, N+1)          — all ones
        """
        import torch

        tok = self.tokenizer
        start_id = tok("<|im_start|>")["input_ids"][0]
        N = len(point_cloud)

        # Dummy input_ids just to drive embed dtype / device placement
        dummy_ids = torch.zeros(
            (1, N + 1), dtype=torch.long, device=self.device,
        )
        dummy_ids[0, -1] = start_id

        pc_t = (
            torch.tensor(point_cloud, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            inputs_embeds = self.model.encode_point_cloud(
                dummy_ids, pc_t, text_start_token_id=start_id,
            )
            attention_mask = torch.ones(
                (1, N + 1), dtype=torch.long, device=self.device,
            )

            eos_id = tok.eos_token_id or tok("<|endoftext|>")["input_ids"][0]
            # Repetition penalty breaks the `w0=cq.Workplane('ZX',origin=(0`
            # loop the model falls into otherwise. The paper does
            # test-time multi-candidate sampling but even their demo
            # with greedy decoding doesn't loop — probably because
            # their training setup sees EOS reliably. On transformers
            # 5.x we need a little help.
            out = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_id,
                do_sample=False,                # greedy (paper default)
                repetition_penalty=1.1,
                no_repeat_ngram_size=8,
            )

        # When inputs_embeds is used, generate() returns only the NEW
        # tokens (it doesn't prepend the input). Decode directly.
        raw = tok.batch_decode(out, skip_special_tokens=False)[0]
        # Trim the <|im_start|>/<|endoftext|> wrappers if present
        if "<|im_start|>" in raw:
            raw = raw.split("<|im_start|>", 1)[1]
        if "<|endoftext|>" in raw:
            raw = raw.split("<|endoftext|>", 1)[0]
        return raw

    # --- Execute generated CadQuery --------------------------------------

    def _execute(self, py_string: str) -> Optional[trimesh.Trimesh]:
        """Exec() the CadQuery script in a sandboxed namespace, tessellate
        the resulting solid, and return a trimesh. Returns None on failure.

        Security note: we restrict globals to ``cq`` + a blank ``__builtins__``
        whitelist, but any ``exec`` of model-generated Python is inherently
        risky. Treat this as a research tool.
        """
        try:
            import cadquery as cq
        except ImportError:
            raise RuntimeError(
                "[cad_recode] cadquery not installed. "
                "pip install cadquery (brings ~400MB OpenCascade)."
            )

        # Restricted globals — cq + a scoped __import__ that only allows
        # ``cadquery`` (or cq alias) and the math stdlib. The model is
        # known to emit ``import cadquery as cq`` as the first line of
        # most scripts, so we need __import__ to function but only for
        # a whitelist.
        _allowed_imports = {"cadquery", "math", "numpy"}
        _real_import = __import__

        def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".")[0] not in _allowed_imports:
                raise ImportError(
                    f"[cad_recode sandbox] import of {name!r} blocked"
                )
            return _real_import(name, globals, locals, fromlist, level)

        safe_builtins = {
            "__import__": _guarded_import,
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "tuple": tuple,
            "dict": dict,
            "set": set,
            "str": str,
            "True": True,
            "False": False,
            "None": None,
            "print": print,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "sum": sum,
            "any": any,
            "all": all,
        }
        ns = {"cq": cq, "cadquery": cq, "__builtins__": safe_builtins}

        try:
            exec(py_string, ns)
        except Exception as e:
            warnings.warn(f"[cad_recode] generated script failed to exec: {e}")
            return None

        if "r" not in ns:
            warnings.warn(
                "[cad_recode] generated script did not set variable 'r'"
            )
            return None

        r = ns["r"]
        try:
            solid = r.val() if hasattr(r, "val") else r
            verts, faces = solid.tessellate(0.001, 0.1)
        except Exception as e:
            warnings.warn(f"[cad_recode] tessellate failed: {e}")
            return None

        # cadquery returns list[cq.Vector]; convert
        v_np = np.asarray([(v.x, v.y, v.z) for v in verts], dtype=np.float32)
        f_np = np.asarray(faces, dtype=np.int64)
        if len(v_np) == 0 or len(f_np) == 0:
            return None
        return trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)

    # --- Scoring ----------------------------------------------------------

    @staticmethod
    def _chamfer(a_pc: np.ndarray, b_pc: np.ndarray) -> float:
        """Symmetric Chamfer between two point sets (both (N, 3))."""
        from scipy.spatial import cKDTree

        tree_a = cKDTree(a_pc)
        tree_b = cKDTree(b_pc)
        d_ab, _ = tree_a.query(b_pc, k=1)
        d_ba, _ = tree_b.query(a_pc, k=1)
        return float(d_ab.mean() + d_ba.mean())

    # --- Public API -------------------------------------------------------

    def refit(
        self,
        mesh: trimesh.Trimesh,
        n_points: int = 256,
        n_candidates: int = 1,
        max_new_tokens: int = 768,
        seed: Optional[int] = 0,
        map_back_to_input_frame: bool = True,
    ) -> CadRecodeResult:
        """Reverse-engineer CadQuery code from ``mesh`` and execute it.

        Args:
            mesh: input trimesh.Trimesh. Will be normalised internally.
            n_points: unused except for an informational warning — model
                is hardcoded to 256. Left in the signature for API clarity.
            n_candidates: number of FPS seeds to try. Each is scored by
                Chamfer distance; the lowest-distance candidate is returned.
                1 = fast (paper default for demo), 10 = paper-quality.
            max_new_tokens: generation cap. 768 covers all paper outputs.
            seed: base random seed. Candidate k uses seed+k.
            map_back_to_input_frame: if True, the returned refit_mesh is
                in the same frame as the input (undoing the cube-of-2
                normalisation). If False, refit_mesh stays normalised.

        Returns:
            CadRecodeResult with CadQuery script, refit mesh, and metadata.
        """
        if n_points != self.N_POINTS:
            warnings.warn(
                f"[cad_recode] n_points={n_points} ignored; model is "
                f"hardcoded to {self.N_POINTS}"
            )

        norm_mesh, centroid, inv_scale = self._normalise(mesh)

        best: Optional[tuple[str, Optional[trimesh.Trimesh], float, np.ndarray]] = None
        candidates: list = []

        for k in range(max(1, n_candidates)):
            pc = self._sample_points(
                norm_mesh, self.N_POINTS, seed=(seed + k) if seed is not None else None
            )
            code = self._generate_one(pc, max_new_tokens=max_new_tokens)
            refit = self._execute(code)

            # Chamfer scoring for ranking
            chamfer = float("inf")
            if refit is not None and len(refit.vertices) > 0:
                refit_pc_v, _ = trimesh.sample.sample_surface(
                    refit, self.N_PRE_POINTS
                )
                chamfer = self._chamfer(pc, refit_pc_v.astype(np.float32))

            candidates.append((code, refit, chamfer, pc))
            if best is None or chamfer < best[2]:
                best = (code, refit, chamfer, pc)

        code, refit, chamfer, pc = best
        candidates.sort(key=lambda c: c[2])

        if refit is not None and map_back_to_input_frame:
            refit = refit.copy()
            refit.apply_scale(inv_scale)
            refit.apply_translation(centroid)

        return CadRecodeResult(
            cadquery_code=code,
            refit_mesh=refit,
            point_cloud=pc,
            input_centroid=centroid,
            input_scale=inv_scale,
            chamfer=chamfer if math.isfinite(chamfer) else None,
            candidates=[(c[0], c[1], c[2]) for c in candidates],
        )
