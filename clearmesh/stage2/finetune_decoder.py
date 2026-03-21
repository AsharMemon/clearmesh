"""Fine-tune TRELLIS.2's FlexiDualGridVaeDecoder with feature-space jitter.

Adapted from UltraShape's decoder fine-tuning recipe for TRELLIS.2's architecture.

UltraShape uses a cross-attention decoder with coordinate-space query jitter:
  - Perturbs encoder query points by U[-1/128, 1/128] in [-1,1] space
  - Trains with SDF supervision at 300K query points
  - 55K steps total (40K at 4096 tokens + 15K at 8192 tokens)

TRELLIS.2 uses a sparse conv decoder (FlexiDualGridVaeDecoder):
  - Integer voxel coordinates (no continuous queries)
  - Sparse 3D convolutions upsample features to high resolution
  - FlexiCubes extracts mesh from upsampled features
  - Position-invariant by design (sparse convs don't memorize positions)

Adaptation: We apply FEATURE-SPACE jitter instead of coordinate jitter.
  - Add uniform noise to SLAT features: SLAT_noisy = SLAT + U[-eps, eps]
  - Train decoder to produce same output from noisy vs clean SLAT
  - Self-distillation: frozen teacher decoder (clean) vs trainable student (noisy)
  - Loss = MSE(student_output, teacher_output) on decoder intermediate features
  - This makes the decoder robust to distribution shift from our refinement DiT

Why feature jitter works for TRELLIS.2:
  The sparse conv decoder is already position-invariant. The real issue is that
  our DiT will produce SLAT features with a slightly different distribution than
  the original diffusion model. Feature jitter during decoder fine-tuning teaches
  the decoder to handle this distribution shift gracefully.

Progressive schedule (55K total, matching UltraShape):
  Phase 1: 40K steps, max 4096 tokens, lr 1e-5
  Phase 2: 15K steps, max 8192 tokens, lr 1e-5

Usage:
    # On Vast.ai pod with TRELLIS.2 installed
    python -m clearmesh.stage2.finetune_decoder --config configs/finetune_decoder.yaml

    # Multi-GPU
    torchrun --nproc_per_node=4 -m clearmesh.stage2.finetune_decoder \
        --config configs/finetune_decoder.yaml
"""

import argparse
import copy
import json
import logging
import math
import os
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SparseTensor utilities
# ---------------------------------------------------------------------------

def _import_sparse():
    """Import TRELLIS.2 sparse tensor utilities (available on pod)."""
    try:
        from trellis2.modules.sparse import basic as sp
        return sp
    except ImportError:
        try:
            from trellis.representations import sparse as sp
            return sp
        except ImportError:
            try:
                import spconv.pytorch as sp
                return sp
            except ImportError:
                raise ImportError(
                    "None of trellis2.modules.sparse, trellis.representations.sparse, "
                    "or spconv found. Run on the Vast.ai pod with TRELLIS.2 installed."
                )


def make_sparse_tensor(feats: torch.Tensor, positions: torch.Tensor, sp_module):
    """Create SparseTensor from features and integer voxel positions.

    Args:
        feats: (N, D) float tensor of SLAT features
        positions: (N, 3) int tensor of voxel coordinates
        sp_module: imported sparse module

    Returns:
        SparseTensor compatible with TRELLIS.2 decoder
    """
    N = feats.shape[0]
    batch_idx = torch.zeros(N, 1, dtype=torch.int32, device=feats.device)
    coords = torch.cat([batch_idx, positions.int().to(feats.device)], dim=1)
    return sp_module.SparseTensor(feats=feats, coords=coords)


# ---------------------------------------------------------------------------
# Feature jitter
# ---------------------------------------------------------------------------

def apply_feature_jitter(
    slat_feats: torch.Tensor,
    jitter_scale: float = 0.1,
    noise_type: str = "uniform",
) -> torch.Tensor:
    """Apply feature-space jitter to SLAT features.

    Equivalent to UltraShape's coordinate query jitter, adapted for TRELLIS.2's
    sparse conv decoder. Instead of perturbing coordinates, we perturb features.

    Args:
        slat_feats: (N, 32) SLAT features
        jitter_scale: Noise magnitude relative to feature std.
            UltraShape uses 1/128 ~ 0.0078 for coord jitter.
            For feature space, we use ~0.05-0.1 as a starting point.
        noise_type: "uniform" (U[-scale, scale]) or "gaussian" (N(0, scale^2))

    Returns:
        (N, 32) jittered SLAT features
    """
    if jitter_scale <= 0:
        return slat_feats

    if noise_type == "uniform":
        noise = (torch.rand_like(slat_feats) - 0.5) * 2 * jitter_scale
    elif noise_type == "gaussian":
        noise = torch.randn_like(slat_feats) * jitter_scale
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return slat_feats + noise


# ---------------------------------------------------------------------------
# Dataset: loads pre-generated SLAT pairs for decoder fine-tuning
# ---------------------------------------------------------------------------

class DecoderFinetuneDataset(Dataset):
    """Load SLAT data for decoder fine-tuning.

    Uses the same data format as generate_slat_pairs.py output:
        uid/
            fine_slat.npy     (N, 32)  — clean SLAT from 1024 model (target)
            coarse_slat.npy   (N, 32)  — coarse SLAT from 512 model (optional)
            positions.npy     (N, 3)   — shared voxel positions

    For decoder fine-tuning, we primarily use fine_slat as the clean reference.
    The decoder should learn to handle perturbations of this clean input.
    """

    def __init__(
        self,
        data_dir: str | list[str],
        max_tokens: int = 4096,
        voxel_dim: int = 32,
    ):
        if isinstance(data_dir, list):
            self.data_dirs = [Path(d) for d in data_dir]
        else:
            self.data_dirs = [Path(data_dir)]
        self.max_tokens = max_tokens
        self.voxel_dim = voxel_dim
        self.pairs = self._discover_pairs()
        logger.info(
            f"DecoderFinetuneDataset: {len(self.pairs)} samples from "
            f"{[str(d) for d in self.data_dirs]}"
        )

    def _discover_pairs(self) -> list[Path]:
        """Find all valid SLAT directories across all data dirs."""
        pairs = []
        search_dirs = []
        for data_dir in self.data_dirs:
            search_dirs.append(data_dir)
            # Also search shard subdirectories
            for shard_dir in sorted(data_dir.glob("shard_*")):
                search_dirs.append(shard_dir)

        for search_dir in search_dirs:
            for d in sorted(search_dir.iterdir()):
                if not d.is_dir():
                    continue
                fine = d / "fine_slat.npy"
                pos = d / "positions.npy"
                if fine.exists() and pos.exists():
                    pairs.append(d)

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        d = self.pairs[idx % len(self.pairs)]

        fine_slat = np.load(d / "fine_slat.npy").astype(np.float32)
        positions = np.load(d / "positions.npy").astype(np.float32)

        N = fine_slat.shape[0]

        # Subsample if too many tokens
        if N > self.max_tokens:
            indices = np.random.choice(N, self.max_tokens, replace=False)
            fine_slat = fine_slat[indices]
            positions = positions[indices]
            N = self.max_tokens

        return {
            "fine_slat": torch.from_numpy(fine_slat),       # (N, 32)
            "positions": torch.from_numpy(positions),        # (N, 3)
            "n_tokens": N,
            "uid": d.name,
        }


def decoder_collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length SLAT samples with padding."""
    max_n = max(b["n_tokens"] for b in batch)
    B = len(batch)
    D = batch[0]["fine_slat"].shape[-1]

    fine_slat = torch.zeros(B, max_n, D)
    positions = torch.zeros(B, max_n, 3)
    mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, b in enumerate(batch):
        n = b["n_tokens"]
        fine_slat[i, :n] = b["fine_slat"]
        positions[i, :n] = b["positions"]
        mask[i, :n] = True

    return {
        "fine_slat": fine_slat,
        "positions": positions,
        "mask": mask,
        "n_tokens": [b["n_tokens"] for b in batch],
        "uids": [b["uid"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Decoder wrapper — adapts TRELLIS.2's decoder for fine-tuning
# ---------------------------------------------------------------------------

class DecoderWrapper(torch.nn.Module):
    """Wraps TRELLIS.2's FlexiDualGridVaeDecoder for fine-tuning.

    This wrapper:
    1. Takes SLAT features + positions as separate tensors
    2. Creates SparseTensor internally
    3. Runs the decoder's forward pass
    4. Returns output features (before FlexiCubes mesh extraction)

    The actual decoder API is discovered at runtime since TRELLIS.2's
    internal structure may vary between versions.
    """

    def __init__(self, decoder_model, sp_module):
        super().__init__()
        self.decoder = decoder_model
        self.sp = sp_module

    def forward(
        self, slat_feats: torch.Tensor, positions: torch.Tensor,
    ) -> torch.Tensor:
        """Run decoder on SLAT features.

        Args:
            slat_feats: (N, 32) SLAT features
            positions: (N, 3) integer voxel positions

        Returns:
            output features from decoder (format depends on TRELLIS.2 version)
        """
        sparse_input = make_sparse_tensor(slat_feats, positions, self.sp)
        output = self.decoder(sparse_input)

        # Extract features from output (SparseTensor or dict)
        if hasattr(output, "feats"):
            return output.feats
        elif hasattr(output, "F"):
            return output.F
        elif isinstance(output, dict):
            # Some decoders return a dict with multiple outputs
            for key in ["feats", "features", "sdf", "logits"]:
                if key in output:
                    return output[key]
        elif isinstance(output, (tuple, list)):
            # Some return (features, coords) tuple
            return output[0] if hasattr(output[0], "shape") else output[0].feats

        raise RuntimeError(
            f"Cannot extract features from decoder output type: {type(output)}. "
            f"Inspect decoder with --inspect flag and adapt this wrapper."
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def setup_distributed():
    """Setup DDP if WORLD_SIZE > 1."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        from datetime import timedelta
        # 30-min timeout to handle slow checkpoint saves (5.4GB writes)
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
    else:
        rank = local_rank = 0
        world_size = 1
    return rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def get_progressive_value(schedule, step, key, default):
    """Get value from progressive schedule at given step."""
    value = default
    for entry in schedule:
        if step >= entry["step"]:
            value = entry.get(key, value)
    return value


# ---------------------------------------------------------------------------
# Decoder inspection (run with --inspect to discover API)
# ---------------------------------------------------------------------------

def inspect_decoder(pipeline):
    """Print decoder model details for API discovery.

    Run this first on the pod to understand the decoder's structure:
        python -m clearmesh.stage2.finetune_decoder --inspect

    Output tells you:
    - Decoder class name and module structure
    - Forward method signature
    - Number of parameters
    - Input/output format
    """
    # Find the decoder model
    decoder = None
    decoder_key = None
    possible_keys = [
        "shape_slat_decoder",
        "slat_decoder_mesh",
        "shape_decoder",
        "decoder",
        "slat_decoder",
    ]

    if hasattr(pipeline, "models") and isinstance(pipeline.models, dict):
        for key in possible_keys:
            if key in pipeline.models:
                decoder = pipeline.models[key]
                decoder_key = key
                break
        if decoder is None:
            print(f"Available model keys: {list(pipeline.models.keys())}")
    else:
        for key in possible_keys:
            if hasattr(pipeline, key):
                decoder = getattr(pipeline, key)
                decoder_key = key
                break

    if decoder is None:
        print("ERROR: Could not find decoder model in pipeline.")
        print(f"Pipeline attributes: {[a for a in dir(pipeline) if not a.startswith('_')]}")
        return

    print(f"\n{'='*60}")
    print(f"TRELLIS.2 Decoder Inspection")
    print(f"{'='*60}")
    print(f"Key:              {decoder_key}")
    print(f"Class:            {decoder.__class__.__name__}")
    print(f"Module:           {decoder.__class__.__module__}")

    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Total params:     {total_params / 1e6:.1f}M")
    print(f"Trainable params: {trainable_params / 1e6:.1f}M")

    print(f"\n--- Top-level modules ---")
    for name, module in decoder.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {module.__class__.__name__} ({n_params/1e6:.1f}M params)")

    print(f"\n--- Forward method ---")
    import inspect
    sig = inspect.signature(decoder.forward)
    print(f"  Signature: forward{sig}")

    # Try a test forward pass
    print(f"\n--- Test forward pass ---")
    sp = _import_sparse()
    N = 100
    test_feats = torch.randn(N, 32, device="cuda", dtype=torch.float32)
    test_coords = torch.randint(0, 32, (N, 3), device="cuda", dtype=torch.int32)
    batch_idx = torch.zeros(N, 1, dtype=torch.int32, device="cuda")
    test_coords_full = torch.cat([batch_idx, test_coords], dim=1)

    try:
        test_input = sp.SparseTensor(feats=test_feats, coords=test_coords_full)
        with torch.no_grad():
            test_output = decoder(test_input)

        if hasattr(test_output, "feats"):
            print(f"  Output type: SparseTensor")
            print(f"  Output feats shape: {test_output.feats.shape}")
            print(f"  Output coords shape: {test_output.coords.shape}")
        elif hasattr(test_output, "F"):
            print(f"  Output type: SparseTensor (F attribute)")
            print(f"  Output F shape: {test_output.F.shape}")
        elif isinstance(test_output, dict):
            print(f"  Output type: dict with keys {list(test_output.keys())}")
            for k, v in test_output.items():
                if hasattr(v, "shape"):
                    print(f"    {k}: {v.shape}")
        elif isinstance(test_output, (tuple, list)):
            print(f"  Output type: {type(test_output).__name__} of length {len(test_output)}")
            for i, v in enumerate(test_output):
                if hasattr(v, "shape"):
                    print(f"    [{i}]: {v.shape}")
                elif hasattr(v, "feats"):
                    print(f"    [{i}]: SparseTensor with feats {v.feats.shape}")
        else:
            print(f"  Output type: {type(test_output)}")
    except Exception as e:
        print(f"  Test forward pass failed: {e}")
        print(f"  This is expected — the decoder may need specific input format.")
        print(f"  Try with actual SLAT data from generate_slat_pairs.py")

    # Check for decode_shape_slat method on pipeline
    print(f"\n--- Pipeline decode method ---")
    if hasattr(pipeline, "decode_shape_slat"):
        sig = inspect.signature(pipeline.decode_shape_slat)
        print(f"  decode_shape_slat{sig}")
    else:
        print(f"  No decode_shape_slat method found")

    # Check pipeline source for decode method
    if hasattr(pipeline, "decode_shape_slat"):
        try:
            source = inspect.getsource(pipeline.decode_shape_slat)
            print(f"\n--- decode_shape_slat source ---")
            # Print first 50 lines
            lines = source.split("\n")[:50]
            for line in lines:
                print(f"  {line}")
            if len(source.split("\n")) > 50:
                print(f"  ... ({len(source.split(chr(10)))} total lines)")
        except (OSError, TypeError):
            print(f"  Could not get source code")

    print(f"\n{'='*60}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DecoderTrainer:
    """Fine-tune TRELLIS.2 decoder with feature-space jitter.

    Self-distillation approach:
      - Teacher: frozen decoder copy (produces reference output from clean SLAT)
      - Student: trainable decoder (produces output from jittered SLAT)
      - Loss: MSE between student and teacher outputs
    """

    def __init__(
        self,
        config: dict,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = is_main_process(rank)

        if world_size > 1:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # --- Load TRELLIS.2 pipeline ---
        if self.is_main:
            print("Loading TRELLIS.2 pipeline...")

        self.sp = _import_sparse()
        self._load_decoder(config)

        # --- DDP ---
        if world_size > 1:
            self.student = DDP(
                self.student, device_ids=[local_rank],
                find_unused_parameters=True,
            )
            if self.is_main:
                print(f"DDP enabled: {world_size} GPUs")

        # --- Parameter counts ---
        if self.is_main:
            raw = self._raw_student()
            total = sum(p.numel() for p in raw.parameters())
            trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
            print(f"Decoder: {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")

        # --- Optimizer ---
        lr = config.get("lr", 1e-5)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=lr,
            betas=tuple(config.get("betas", [0.9, 0.99])),
            weight_decay=config.get("weight_decay", 0.01),
        )

        # --- LR scheduler (cosine with warmup) ---
        total_steps = config.get("total_steps", 55_000)
        warmup_steps = config.get("warmup_steps", 500)
        self.warmup_steps = warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

        # --- Jitter config ---
        self.jitter_scale = config.get("jitter_scale", 0.1)
        self.noise_type = config.get("noise_type", "uniform")

        # --- Progressive schedule ---
        self.progressive = config.get("progressive_schedule", [
            {"step": 0, "num_tokens": 4096},
            {"step": 40000, "num_tokens": 8192},
        ])

        # --- Training state ---
        self.global_step = 0
        self.epoch = 0
        self._checkpoint_loaded = False
        self.output_dir = Path(config["output_dir"])
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Preemption handling
        self._emergency_save = False
        signal.signal(signal.SIGUSR1, self._handle_preemption)

        # WandB
        self.wandb = None
        if config.get("use_wandb", False) and self.is_main:
            import wandb
            wandb.init(
                project="clearmesh",
                config=config,
                name="decoder-finetune",
            )
            self.wandb = wandb

    def _load_decoder_directly(self, pipeline_path):
        """Load decoder directly from checkpoint files, bypassing full pipeline.

        This is a fallback for when the full pipeline can't load (e.g. due to
        rembg/DINOv3 incompatibilities with newer PyTorch versions).
        We only need the shape decoder for fine-tuning.
        """
        import json
        import os
        from pathlib import Path

        pipeline_json = os.path.join(pipeline_path, "pipeline.json")
        if not os.path.exists(pipeline_json):
            return None

        with open(pipeline_json, 'r') as f:
            config = json.load(f)

        model_paths = config.get('args', {}).get('models', {})
        decoder_key = 'shape_slat_decoder'
        if decoder_key not in model_paths:
            if self.is_main:
                print(f"No '{decoder_key}' in pipeline.json models: {list(model_paths.keys())}")
            return None

        decoder_rel_path = model_paths[decoder_key]
        decoder_path = os.path.join(pipeline_path, decoder_rel_path)

        try:
            from trellis2 import models as trellis_models
            decoder = trellis_models.from_pretrained(decoder_path)
            if self.is_main:
                print(f"Loaded decoder directly from {decoder_rel_path}")

            # Create a minimal pipeline-like object with .models dict
            class MinimalPipeline:
                def __init__(self, models_dict):
                    self.models = models_dict

            return MinimalPipeline({decoder_key: decoder})
        except Exception as e:
            if self.is_main:
                print(f"Direct decoder loading failed: {e}")
            return None

    def _load_decoder(self, config):
        """Load decoder from TRELLIS.2 pipeline or standalone checkpoint."""
        pipeline_path = config.get(
            "pipeline_path", "/workspace/models/trellis2-4b"
        )

        # Load TRELLIS.2 pipeline (try multiple import paths)
        # First try full pipeline, then fall back to loading decoder directly
        pipeline = None
        for pipeline_cls_path in [
            ("trellis2.pipelines", "Trellis2ImageTo3DPipeline"),
            ("trellis.pipelines", "TreLLiS2Pipeline"),
            ("trellis.pipelines", "TrellisImageTo3DPipeline"),
        ]:
            try:
                import importlib
                mod = importlib.import_module(pipeline_cls_path[0])
                PipelineCls = getattr(mod, pipeline_cls_path[1])
                pipeline = PipelineCls.from_pretrained(pipeline_path)
                if self.is_main:
                    print(f"Loaded pipeline: {pipeline_cls_path[1]}")
                break
            except (ImportError, AttributeError):
                continue
            except Exception as e:
                if self.is_main:
                    print(f"Pipeline load failed ({pipeline_cls_path[1]}): {e}")
                    print("Falling back to direct decoder loading...")
                break

        # Fallback: load decoder directly from checkpoint files
        # This bypasses rembg/DINOv3 loading issues on newer PyTorch
        if pipeline is None:
            pipeline = self._load_decoder_directly(pipeline_path)

        if pipeline is None:
            raise RuntimeError(
                f"Could not load TRELLIS.2 pipeline from {pipeline_path}. "
                "Check that the trellis2 package is installed."
            )

        # Extract decoder — TRELLIS.2 uses 'shape_slat_decoder'
        decoder = None
        for key in [
            "shape_slat_decoder",
            "slat_decoder_mesh",
            "shape_decoder",
            "decoder",
        ]:
            if hasattr(pipeline, "models") and isinstance(pipeline.models, dict):
                if key in pipeline.models:
                    decoder = pipeline.models[key]
                    if self.is_main:
                        print(f"Found decoder at models['{key}']")
                    break

        if decoder is None:
            raise RuntimeError(
                "Could not find decoder in pipeline. "
                f"Available keys: {list(pipeline.models.keys()) if hasattr(pipeline, 'models') else 'N/A'}. "
                "Run with --inspect to see available models."
            )

        # Convert decoder to float32 for training stability.
        # fp16 blocks cause NaN gradients during backward pass.
        # Note: spconv's CUDA kernels don't support fp16/bf16, so
        # mixed precision is not available. Use multi-GPU DDP for speed.
        if hasattr(decoder, "convert_to_fp32"):
            decoder.convert_to_fp32()
            decoder.use_fp16 = False
            decoder.dtype = torch.float32
            if self.is_main:
                print("Converted decoder to float32 for training stability")

        # Student: trainable decoder
        self.student = decoder.to(self.device)
        self.student.train()

        # Teacher: frozen copy
        self.teacher = copy.deepcopy(decoder).to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Store pipeline reference for decode_shape_slat access
        self._pipeline = pipeline

        if self.is_main:
            n_params = sum(p.numel() for p in decoder.parameters())
            n_train = sum(
                p.numel() for p in decoder.parameters() if p.requires_grad
            )
            print(f"Decoder loaded: {decoder.__class__.__name__}")
            print(f"Decoder: {n_params/1e6:.1f}M total, {n_train/1e6:.1f}M trainable")

    def _raw_student(self):
        return self.student.module if isinstance(self.student, DDP) else self.student

    def _handle_preemption(self, signum, frame):
        print(f"\n!!! PREEMPTION (rank {self.rank}) — saving emergency checkpoint !!!")
        self._emergency_save = True

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _build_dataloader(self, max_tokens: int) -> DataLoader:
        batch_size = self.config.get("batch_size", 4)
        dataset = DecoderFinetuneDataset(
            data_dir=self.config["data_dir"],
            max_tokens=max_tokens,
        )

        sampler = None
        shuffle = True
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=decoder_collate_fn,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: str = "latest"):
        if not self.is_main:
            return
        raw = self._raw_student()

        # For milestone saves, save lightweight weights FIRST (fast, 1.8GB)
        # so we don't lose progress if the full save (5.4GB) gets interrupted
        if tag not in ("latest",):
            decoder_only = {"decoder": raw.state_dict(), "step": self.global_step}
            torch.save(
                decoder_only,
                self.output_dir / f"decoder_{tag}_weights.pt",
            )
            print(f"  Decoder weights: decoder_{tag}_weights.pt")

        # Save full checkpoint with optimizer state for resume (slow, 5.4GB)
        # Write to temp file then rename for atomicity
        latest_path = self.output_dir / "decoder_latest.pt"
        tmp_path = self.output_dir / "decoder_latest.pt.tmp"
        ckpt = {
            "decoder": raw.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config,
        }
        torch.save(ckpt, tmp_path)
        tmp_path.rename(latest_path)  # atomic rename
        print(f"Decoder checkpoint saved: {latest_path} (step {self.global_step})")

    def load_checkpoint(self, path: str | None = None):
        p = Path(path) if path else self.output_dir / "decoder_latest.pt"

        # If latest.pt is missing or corrupted, fall back to newest weights file
        if not p.exists() or p.stat().st_size == 0:
            if self.is_main and p.exists() and p.stat().st_size == 0:
                print(f"WARNING: {p} is corrupted (0 bytes), looking for weight files...")
            weights_files = sorted(
                self.output_dir.glob("decoder_step_*_weights.pt"),
                key=lambda f: int(f.stem.split("_")[2]),  # extract step number
            )
            if weights_files:
                p = weights_files[-1]  # newest weights file
                if self.is_main:
                    print(f"Falling back to weights-only checkpoint: {p}")
            else:
                if self.is_main:
                    print("No decoder checkpoint found, starting from scratch.")
                return False

        if self.is_main:
            print(f"Resuming from {p}")
        ckpt = torch.load(p, map_location=self.device, weights_only=False)
        raw = self._raw_student()
        raw.load_state_dict(ckpt["decoder"])

        # Full checkpoint: restore optimizer, scheduler, step
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.global_step = ckpt["global_step"]
            self.epoch = ckpt["epoch"]
        else:
            # Weights-only: restore step but reset optimizer/scheduler
            self.global_step = ckpt.get("step", 0)
            self.epoch = 0
            if self.is_main:
                print(f"  Weights-only resume — optimizer/scheduler reset at step {self.global_step}")

        self._checkpoint_loaded = True
        if self.is_main:
            print(f"Resumed at step {self.global_step}, epoch {self.epoch}")
        return True

    # ------------------------------------------------------------------
    # Forward pass: decoder with guided subdivisions
    # ------------------------------------------------------------------

    def _decoder_forward_guided(
        self,
        model,
        sparse_input,
        guide_subs=None,
    ):
        """Forward through decoder with optional guided subdivision decisions.

        When guide_subs is None (teacher mode):
            Runs normally, collects subdivision decisions from each
            upsampling block (SparseResBlockC2S3d).

        When guide_subs is provided (student mode):
            Temporarily monkey-patches each upsampling block's to_subdiv
            to return teacher's subdivision instead of computing its own.
            This ensures identical spatial structure between teacher and
            student outputs, enabling aligned MSE comparison.

        The monkey-patching approach correctly handles all block types
        (SparseResBlockC2S3d, SparseResBlockUpsample3d, SparseResBlock3d)
        since it uses each block's original _forward logic.

        Args:
            model: The decoder model (SparseUnetVaeDecoder subclass)
            sparse_input: SparseTensor input
            guide_subs: List of subdivision SparseTensors from teacher

        Returns:
            (output_feats, subs_list)
        """
        h = model.from_latent(sparse_input)
        h = h.type(model.dtype)
        subs = []
        sub_idx = 0

        for i, res in enumerate(model.blocks):
            for j, block in enumerate(res):
                is_upsample = (
                    i < len(model.blocks) - 1 and j == len(res) - 1
                )
                if is_upsample:
                    if guide_subs is not None:
                        # Student mode: monkey-patch to_subdiv to return
                        # teacher's subdivision, then call block normally.
                        # Use object.__setattr__ to bypass torch.nn.Module
                        # type checking on registered child modules.
                        teacher_sub = guide_subs[sub_idx]
                        sub_idx += 1
                        original_to_subdiv = block.to_subdiv
                        object.__setattr__(
                            block, "to_subdiv",
                            lambda x, _s=teacher_sub: _s,
                        )
                        try:
                            h, _ = block(h)
                        finally:
                            object.__setattr__(
                                block, "to_subdiv", original_to_subdiv,
                            )
                    else:
                        # Teacher mode: block computes its own subdivisions
                        h, sub = block(h)
                        subs.append(sub)
                else:
                    h = block(h)

        h = h.type(sparse_input.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = model.output_layer(h)
        return h.feats, subs

    def _make_sparse(self, slat_feats, positions, model):
        """Create SparseTensor with correct dtype for model."""
        model_dtype = next(model.parameters()).dtype
        feats = slat_feats.to(dtype=model_dtype)
        return make_sparse_tensor(feats, positions, self.sp)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict[str, float]:
        """Single training step with feature jitter self-distillation.

        For each sample in batch:
        1. Clean SLAT → teacher decoder → reference features + subs
        2. Jittered SLAT → student decoder (guided by teacher subs) → features
        3. Loss = MSE(student_features, teacher_features)

        Using teacher's subdivision decisions ensures:
        - Same spatial structure → outputs are aligned
        - No empty SparseTensor crashes from jittered features
        - Student learns to produce correct features given spatial structure
        """
        fine_slat = batch["fine_slat"].to(self.device)    # (B, N, 32)
        positions = batch["positions"].to(self.device)     # (B, N, 3)
        mask = batch["mask"].to(self.device)               # (B, N)
        B = fine_slat.shape[0]

        total_loss = torch.tensor(0.0, device=self.device)
        n_valid = 0

        for i in range(B):
            # Get valid tokens for this sample
            valid = mask[i]
            slat_i = fine_slat[i, valid]       # (Ni, 32)
            pos_i = positions[i, valid]         # (Ni, 3)

            if slat_i.shape[0] == 0:
                continue

            # Teacher: clean decode (frozen, eval mode, no grad)
            # Get reference features AND subdivision decisions
            self.teacher.eval()
            with torch.no_grad():
                teacher_sparse = self._make_sparse(slat_i, pos_i, self.teacher)
                teacher_out, teacher_subs = self._decoder_forward_guided(
                    self.teacher, teacher_sparse, guide_subs=None
                )

            # Apply feature jitter
            slat_jittered = apply_feature_jitter(
                slat_i,
                jitter_scale=self.jitter_scale,
                noise_type=self.noise_type,
            )

            # Student: jittered decode using teacher's subdivision structure
            raw_student = self._raw_student()
            raw_student.train()
            student_sparse = self._make_sparse(
                slat_jittered, pos_i, raw_student
            )
            student_out, _ = self._decoder_forward_guided(
                raw_student, student_sparse, guide_subs=teacher_subs
            )

            # MSE loss on output features (cast to float32 for stable loss)
            student_f32 = student_out.float()
            teacher_f32 = teacher_out.detach().float()

            # Both should have same shape since we use same subdivisions
            loss_i = F.mse_loss(student_f32, teacher_f32)

            # Skip NaN losses (safety net)
            if torch.isnan(loss_i) or torch.isinf(loss_i):
                self._nan_count = getattr(self, "_nan_count", 0) + 1
                if self.is_main and self._nan_count <= 5:
                    print(
                        f"  [nan #{self._nan_count}] "
                        f"student range: [{student_f32.min():.2f}, {student_f32.max():.2f}], "
                        f"teacher range: [{teacher_f32.min():.2f}, {teacher_f32.max():.2f}]"
                    )
                continue

            total_loss = total_loss + loss_i
            n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid
            total_loss.backward()
        # else: all samples skipped — skip backward

        return {"mse": total_loss.item(), "n_valid": n_valid}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        total_steps = self.config.get("total_steps", 55_000)
        save_interval = self.config.get("save_interval", 5000)
        log_interval = self.config.get("log_interval", 50)
        grad_accum = self.config.get("grad_accum_steps", 1)

        if not self._checkpoint_loaded:
            self.load_checkpoint()

        # Current progressive values
        cur_tokens = get_progressive_value(
            self.progressive, self.global_step, "num_tokens", 4096
        )
        dataloader = self._build_dataloader(cur_tokens)

        if self.is_main:
            print(f"\n{'='*60}")
            print(f"=== Decoder Fine-tuning with Feature Jitter ===")
            print(f"  Total steps:     {total_steps}")
            print(f"  Current step:    {self.global_step}")
            print(f"  Dataset size:    {len(dataloader.dataset)}")
            print(f"  Batch size:      {self.config.get('batch_size', 4)}")
            print(f"  Grad accum:      {grad_accum}")
            print(f"  World size:      {self.world_size} GPU(s)")
            print(f"  Jitter scale:    {self.jitter_scale}")
            print(f"  Noise type:      {self.noise_type}")
            print(f"  LR:              {self.config.get('lr', 1e-5)}")
            print(f"  Progressive:     {self.progressive}")
            print(f"{'='*60}\n")

        pbar = tqdm(
            total=total_steps,
            initial=self.global_step,
            desc="Decoder FT",
            disable=not self.is_main,
        )

        accum_loss = 0.0
        accum_count = 0

        while self.global_step < total_steps:
            if hasattr(dataloader, "sampler") and isinstance(
                dataloader.sampler, DistributedSampler
            ):
                dataloader.sampler.set_epoch(self.epoch)

            for batch in dataloader:
                if self.global_step >= total_steps:
                    break

                # Emergency save
                if self._emergency_save:
                    self.save_checkpoint(f"emergency_{self.global_step}")
                    if self.is_main:
                        print("Emergency checkpoint saved. Exiting.")
                    if self.world_size > 1:
                        dist.destroy_process_group()
                    sys.exit(0)

                # Progressive schedule: rebuild dataloader if changed
                new_tokens = get_progressive_value(
                    self.progressive, self.global_step, "num_tokens", 4096
                )
                if new_tokens != cur_tokens:
                    if self.is_main:
                        print(
                            f"\n>>> Progressive: tokens {cur_tokens}→{new_tokens} "
                            f"at step {self.global_step}"
                        )
                    cur_tokens = new_tokens
                    dataloader = self._build_dataloader(cur_tokens)
                    break  # restart epoch with new dataloader

                # Train step — catch degenerate samples (empty voxels)
                try:
                    losses = self.train_step(batch)
                except (IndexError, RuntimeError) as e:
                    if self.is_main:
                        logger.warning(
                            f"Step {self.global_step}: skipping bad sample ({e})"
                        )
                    self.optimizer.zero_grad()
                    n_skips += 1
                    self.global_step += 1
                    continue
                accum_loss += losses["mse"]
                accum_count += 1

                # Optimizer step
                if accum_count >= grad_accum:
                    # Only clip/step if at least one valid sample produced grads
                    has_grads = any(
                        p.grad is not None
                        for p in self.student.parameters()
                        if p.requires_grad
                    )
                    if has_grads:
                        torch.nn.utils.clip_grad_norm_(
                            self.student.parameters(),
                            self.config.get("max_grad_norm", 1.0),
                        )
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    pbar.update(1)

                    avg_loss = accum_loss / accum_count
                    accum_loss = 0.0
                    accum_count = 0

                    # Logging
                    if self.is_main and self.global_step % log_interval == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        skip_total = getattr(self, "_skip_count", 0)
                        pbar.set_postfix_str(
                            f"mse: {avg_loss:.6f} | lr: {lr:.2e} | "
                            f"valid: {losses.get('n_valid', '?')} | skips: {skip_total}"
                        )
                        if self.wandb:
                            self.wandb.log({
                                "loss/mse": avg_loss,
                                "lr": lr,
                                "tokens": cur_tokens,
                            }, step=self.global_step)

                    # Checkpoint — barrier before AND after so non-main
                    # ranks wait while rank 0 writes (avoids NCCL timeout)
                    if self.global_step % save_interval == 0:
                        if self.world_size > 1:
                            dist.barrier()  # sync before save
                        self.save_checkpoint(f"step_{self.global_step}")
                        if self.world_size > 1:
                            dist.barrier()  # sync after save

            self.epoch += 1

        pbar.close()
        if self.world_size > 1:
            dist.barrier()  # sync before final save
        self.save_checkpoint("final")
        if self.world_size > 1:
            dist.barrier()  # sync after final save
        if self.is_main:
            print(f"\nDecoder fine-tuning complete at step {self.global_step}")
        if self.world_size > 1:
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune TRELLIS.2 decoder with feature jitter"
    )
    parser.add_argument("--config", help="YAML config path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument(
        "--inspect", action="store_true",
        help="Inspect decoder architecture and exit (run first!)"
    )
    parser.add_argument(
        "--pipeline_path", default="JeffreyXiang/TRELLIS-image-large",
        help="TRELLIS.2 model path (for --inspect)"
    )
    args = parser.parse_args()

    if args.inspect:
        # Inspection mode: discover decoder API
        print("Loading pipeline for inspection...")
        for cls_path in [
            ("trellis2.pipelines", "Trellis2ImageTo3DPipeline"),
            ("trellis.pipelines", "TreLLiS2Pipeline"),
            ("trellis.pipelines", "TrellisImageTo3DPipeline"),
        ]:
            try:
                import importlib
                mod = importlib.import_module(cls_path[0])
                PipelineCls = getattr(mod, cls_path[1])
                pipeline = PipelineCls.from_pretrained(args.pipeline_path)
                break
            except (ImportError, AttributeError):
                continue
        else:
            print("ERROR: Could not load any TRELLIS.2 pipeline class")
            return
        pipeline.cuda()
        inspect_decoder(pipeline)
        return

    if not args.config:
        parser.error("--config is required unless using --inspect")

    rank, local_rank, world_size = setup_distributed()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.output_dir:
        config["output_dir"] = args.output_dir

    trainer = DecoderTrainer(
        config, rank=rank, local_rank=local_rank, world_size=world_size
    )
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    trainer.train()


if __name__ == "__main__":
    main()
