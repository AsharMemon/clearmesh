# CAD-Recode Runtime API — Research Notes

Research target: the exact runtime API of `filaPro/cad-recode` so we can write a
ClearMesh wrapper that takes a `trimesh.Trimesh` and returns `(cadquery_code, refit_mesh)`.

Paper: *CAD-Recode: Reverse Engineering CAD Code from Point Clouds*, Rukhovich et
al., ICCV 2025. arXiv 2412.14042.

---

## 1. Repo / License / Weights

| Item | Value |
| --- | --- |
| GitHub | https://github.com/filaPro/cad-recode |
| Stars | ~220 (Apr 2026) |
| Last commit | March 16, 2025 (v1.5 release) |
| Repo license | `LICENSE.md` present (check file before commercial use) |
| **Weights license** | **CC-BY-NC-4.0** (Creative Commons NonCommercial) — RESEARCH ONLY |
| Weights v1 | https://huggingface.co/filapro/cad-recode |
| Weights v1.5 (latest) | https://huggingface.co/filapro/cad-recode-v1.5 |
| HF Space demo | https://huggingface.co/spaces/filapro/cad-recode |
| Project page | https://cad-recode.github.io/ |
| Model size | ~1.5B params (Qwen2-1.5B + Fourier projector + 1 linear layer). F32+BF16 safetensors, roughly 3 GB. Paper ablation calls it "1.5 B" in Table 4. |

**Important licence caveat**: CC-BY-NC-4.0 means the published weights cannot
be used for any commercial product. For ClearMesh internal research / ablation
experiments this is fine; for anything shipped we'd either need to retrain from
scratch on a commercial-compatible dataset, or contact the authors.

---

## 2. Install commands (verbatim behaviour from `Dockerfile`)

The repo ships a Dockerfile rather than a `requirements.txt`. Rough reproduction:

```bash
# Base: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
# apt deps:
apt-get install -y git wget libgl1-mesa-glx libosmesa6-dev libglu1-mesa-dev

# pip deps (non-exhaustive; key pinned versions):
pip install numpy==2.2.0 scipy==1.14.1
pip install transformers==4.47.1
pip install trimesh==4.5.3
pip install flash-attn==2.7.2.post1 --no-build-isolation

# Git-based installs (these are the ones that hurt):
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/CadQuery/cadquery.git@<commit>

# Open3D built from source for headless rendering (only needed for the demo
# notebook's viz; not for pure inference).
```

**Gotchas**:
- `pytorch3d` needs to compile against CUDA 12.4 + torch 2.5.1 — expect
  5-15 minutes of nvcc on first install and painful wheel-hunting otherwise.
- `flash-attn` wheel must match torch/CUDA exactly.
- CadQuery brings in OpenCascade via `cadquery-ocp` — roughly 400 MB.
- Open3D from source is ONLY needed for the notebook's rendering. For headless
  inference we can skip it entirely.

**Minimum viable deps for inference only**:
`torch, transformers, pytorch3d, cadquery, trimesh, numpy`.

---

## 3. Minimal inference snippet

Distilled from `demo.ipynb`. This is the exact API, not a rewrite.

```python
import torch
import numpy as np
import trimesh
import cadquery as cq
from torch import nn
from transformers import AutoTokenizer, Qwen2ForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from pytorch3d.ops import sample_farthest_points


class FourierPointEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        frequencies = 2.0 ** torch.arange(8, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        # 3 raw coords + 3 * 2 * 8 sin/cos = 51 -> hidden
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points):
        x = (points.unsqueeze(-1) * self.frequencies).view(*points.shape[:-1], -1)
        x = torch.cat((points, x.sin(), x.cos()), dim=-1)
        return self.projection(x)


class CADRecode(Qwen2ForCausalLM):
    # Full class lives in demo.ipynb; extends Qwen2ForCausalLM with a
    # FourierPointEncoder and overrides forward() to splice point embeddings
    # into the input embedding stream wherever attention_mask == -1.
    ...


def mesh_to_point_cloud(mesh, n_points=256, n_pre_points=8192):
    verts, _ = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(
        torch.tensor(verts).unsqueeze(0).float(), K=n_points
    )
    return np.asarray(verts[ids[0].numpy()])


# Load
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-1.5B", pad_token="<|im_end|>", padding_side="left"
)
model = CADRecode.from_pretrained(
    "filapro/cad-recode-v1.5",
    torch_dtype="auto",
    attn_implementation="flash_attention_2",  # or "sdpa"
).to("cuda").eval()

# Prep input
mesh.apply_translation(-mesh.bounding_box.centroid)
mesh.apply_scale(2.0 / max(mesh.extents))  # fit into cube of side 2
point_cloud = mesh_to_point_cloud(mesh, n_points=256)  # shape (256, 3)

# input_ids/attention_mask trick: 256 pad tokens with mask=-1 tell CADRecode
# to splice the projected point embeddings in at those positions, followed by
# the real <|im_start|> token.
input_ids = [tokenizer.pad_token_id] * len(point_cloud) + [
    tokenizer("<|im_start|>")["input_ids"][0]
]
attention_mask = [-1] * len(point_cloud) + [1]

with torch.no_grad():
    out = model.generate(
        input_ids=torch.tensor(input_ids).unsqueeze(0).cuda(),
        attention_mask=torch.tensor(attention_mask).unsqueeze(0).cuda(),
        point_cloud=torch.tensor(point_cloud, dtype=torch.float32)
            .unsqueeze(0).cuda(),
        max_new_tokens=768,
        pad_token_id=tokenizer.pad_token_id,
    )

# Decode — take everything between <|im_start|> and <|endoftext|>
raw = tokenizer.batch_decode(out)[0]
py_string = raw.split("<|im_start|>")[1].split("<|endoftext|>")[0]

# Execute → CadQuery solid → mesh
exec(py_string, globals())
compound = globals()["r"].val()                # convention: script assigns to `r`
verts, faces = compound.tessellate(0.001, 0.1) # tol=0.001, angular=0.1
refit_mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], faces)
```

Key non-obvious details:

- **Note `n_points=256`, NOT 16384** — the paper and demo both use 256. The
  ablation (Table 4) shows 256 > 128 in IoU and there is no "more is better"
  axis tested. The 8192 in the pipeline is the mesh-surface pre-sample before
  FPS; it's not what the model sees.
- The model uses a **custom `generate` signature with a `point_cloud=` kwarg** —
  it does not inherit the vanilla HF generate API cleanly. The model class
  monkey-patches generation so that the `attention_mask == -1` positions get
  replaced with Fourier-projected point embeddings.
- The generated script expects the CadQuery workplane to be stored in a global
  `r`. Hardcoded convention, not configurable.
- Test-time sampling in the paper draws 10 candidates with different FPS seeds,
  scores each by Chamfer distance against the input cloud, and picks the best.
  Demo uses a single sample.

---

## 4. Input format spec

| Field | Spec |
| --- | --- |
| dtype | `float32` |
| shape | `(1, 256, 3)` — batched, single element is standard |
| coordinate frame | centred at origin, fit into a cube of edge 2 (i.e. roughly `[-1, 1]^3`) |
| sampling | Farthest-Point-Sampling 256 points from 8192 uniform surface samples |
| normals | NOT used |
| training noise | Gaussian N(0, 0.01) added to 50% of points during training — model is robust to this level of noise |

The paper (§4.3) confirms `n_p = 256`, `d_q = 1536`, `d_p = 3`, Fourier encoding
with 8 frequency bands (= 51-dim input to the linear projector).

---

## 5. Output format spec

| Field | Spec |
| --- | --- |
| Raw | token ids from Qwen2 tokenizer, prefixed with `<|im_start|>` and ending with `<|endoftext|>` |
| Decoded | Python string, valid CadQuery code |
| Length | `max_new_tokens=768` cap; typical output ~50-300 tokens based on the figure-2 examples |
| Convention | Script imports `cadquery as cq`, builds a workplane, assigns the final solid to variable `r` |
| Invalidity ratio | **0.4% (DeepCAD) / 0.5% (Fusion360) / 0.3% (CC3D)** — basically always executes |
| Supported ops | `.box(...)`, `.cylinder(...)`, `.sketch().circle().rectangle().finalize().extrude(...)`, `.union()`. Sketch primitives: lines, arcs, circles. Abstractions: boxes, cylinders, rectangles. |
| **Not supported** | Revolution, fillet, chamfer, loft, sweep — the paper explicitly lists these as out-of-scope for v1 (§ "Results & Analysis"). |

Typical output shape (from Fig 2c in paper):

```python
import cadquery as cq
w = cq.Workplane("XY")
w.box(100, 100, 14).union(
    w.sketch().circle(30, mode="s").finalize().extrude(-10)
)
```

A couple of lines, single-digit operations. Nothing like a real human CAD
script.

---

## 6. Performance numbers

The paper does NOT publish wall-clock latency or VRAM. What we can infer:

- **Model size**: 1.5 B params × bf16 ≈ 3 GB weights + point projector (tiny).
  Plus KV cache for ~1000 tokens. Realistic peak VRAM: 4-6 GB on H100 for
  batch=1. Definitely fits on a single consumer 12 GB GPU.
- **Latency**: Qwen2-1.5B at ~768 output tokens with flash-attn on an H100 is
  typically 1-3 seconds for a single sample. With 10-candidate test-time
  sampling this becomes 10-30 seconds per mesh. The HF ZeroGPU demo Space
  confirms it runs interactively.
- **Quality metrics** (Table 1 & 2, v1 paper, 1M-sample training):
  - DeepCAD: Mean CD 0.30, Median CD 0.16, IoU 92.0%, IR 0.4%
  - Fusion360: Mean CD 0.35, Median CD 0.15, IoU 87.8%, IR 0.5%
  - **Real-world CC3D (scanned, noisy, missing parts): Mean CD 0.76, Median CD 0.31, IoU 74.2%, IR 0.3%** — this is the most relevant number for us.

---

## 7. Failure modes (from paper §5.1 "Results & Analysis" + Fig 5)

Direct paper quotes summarised:

1. **Limited op vocabulary**: "CAD-Recode still lacks the expressiveness to
   model complex shapes that contain operations beyond the extrusion operation
   such as revolution and fillet." Filleted edges, chamfers, swept profiles,
   lofts all get approximated as polygonal extrusions.
2. **Procedural training bias**: trained on 1M procedurally generated shapes
   which are combinations of boxes / cylinders / rectangles / lines / arcs /
   circles — organic shapes are wildly OOD.
3. **CC3D regime is the best proxy for us**: CC3D is real 3D scans with surface
   noise, smoothed edges, missing parts. IoU drops from 92 → 74%, Chamfer
   roughly 2.5× worse. This is the *good* case — CC3D is still mostly prismatic
   mechanical parts.
4. **TRELLIS/Hunyuan regime (organic meshes, statues, toys, creatures)**:
   NOT benchmarked in the paper. Almost certainly degrades severely — there is
   no training signal for smooth curvature or non-axis-aligned freeform
   surfaces. Expect the model to approximate them as a stack of boxes and
   cylinders or produce a near-empty script.
5. **Resolution mismatch**: input is always 256 points. Our meshes have ~2M
   faces. FPS gives reasonable coverage for prismatic parts but will miss fine
   organic details and small protrusions entirely.
6. **Normalisation is critical**: model expects cube-of-2 framing. Any other
   scale breaks the learned geometric prior completely.

Concrete prediction for ClearMesh's use cases:
- **Mechanical TRELLIS output (gears, brackets, enclosures)**: decent — expect
  CC3D-tier quality, IoU ~70-75% of the prismatic skeleton.
- **Organic TRELLIS output (characters, plants, statues)**: poor — expect the
  model to return a box or a stack of cylinders vaguely bounding the subject.
- **Hybrid meshes**: model will preserve the prismatic parts and collapse the
  organic parts.

---

## 8. Proposed ClearMesh wrapper API

```python
# clearmesh/refit/cad_recode.py
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import trimesh


@dataclass
class CadRecodeResult:
    cadquery_code: str           # raw Python string
    refit_mesh: trimesh.Trimesh  # tessellated CadQuery solid
    point_cloud: np.ndarray      # (256, 3) the model actually saw
    chamfer: Optional[float]     # vs. input point cloud, if scored
    candidates: Optional[list]   # other sampled candidates when k>1


class CadRecodeRefiner:
    """Wrap filapro/cad-recode-v1.5 for ClearMesh.

    Licence: model is CC-BY-NC-4.0. RESEARCH ONLY.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "filapro/cad-recode-v1.5",
        attn_implementation: str = "flash_attention_2",
    ):
        # load tokenizer, CADRecode (custom class), cache on self
        ...

    def refit(
        self,
        mesh: trimesh.Trimesh,
        n_points: int = 256,     # fixed by model; keep kwarg for API clarity
        n_candidates: int = 1,   # 10 for paper-quality test-time sampling
        max_new_tokens: int = 768,
        tessellation_tol: float = 0.001,
        seed: Optional[int] = None,
    ) -> CadRecodeResult:
        """Reverse-engineer CadQuery code from `mesh` and execute it.

        Note: n_points != the 16384 in our sketch signature; the model is
        hardcoded to 256. If callers pass 16384 we silently downsample via FPS.
        """
        ...

    def _normalise(self, mesh):
        """Center at origin, scale so max extent = 2."""
        ...

    def _sample_points(self, mesh, n_points, seed):
        """Surface sample 8192 + FPS to n_points."""
        ...

    def _execute(self, py_string) -> trimesh.Trimesh:
        """exec() the CadQuery code in an isolated namespace, pull var `r`,
        tessellate, return trimesh."""
        ...

    def _score(self, mesh, candidate_mesh) -> float:
        """Symmetric Chamfer for candidate re-ranking."""
        ...
```

**API recommendations for the wrapper**:
- **Override the `n_points=16384` kwarg from the sketch** to 256 internally.
  Log a warning when the caller passes anything else. The model physically
  cannot accept more.
- **Do normalisation INSIDE the wrapper** and keep the original transform so
  we can return the refit mesh in the caller's frame. The paper's bounding-box
  cube-of-2 is not optional.
- **Sandbox `exec()`**. The model is generating Python that we run. Even
  without malicious intent a bad generation can `import os; os.system(...)`.
  At minimum, `exec(code, {"cq": cq, "cadquery": cq})` in a fresh globals dict.
- **Expose test-time sampling as `n_candidates`**. The paper uses 10 and it
  reduces invalidity to <1%. Set default 1 for speed, 10 for quality.
- **Return the raw `cadquery_code` string** as the primary artefact — it's
  arguably more valuable than the mesh (editable, interpretable, diffable).
- **Cache model weights and avoid reloading on every call**. Load once per
  process.

---

## References

- Paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Rukhovich_CAD-Recode_Reverse_Engineering_CAD_Code_from_Point_Clouds_ICCV_2025_paper.pdf
- arXiv: https://arxiv.org/abs/2412.14042
- GitHub: https://github.com/filaPro/cad-recode
- HF v1.5: https://huggingface.co/filapro/cad-recode-v1.5
- Demo space: https://huggingface.co/spaces/filapro/cad-recode
