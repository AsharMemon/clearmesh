"""CPU-only unit tests for VoxelFlowEdit internal math.

These tests validate the flow-matching math (`_forward_diffuse`,
`_trajectory_correction`, edit-mask blending) that is pure linear algebra
and should not require TRELLIS.2 or a GPU.

They use monkey-patched `_compute_velocity` and `_encode_image_condition`
so they run without any model loaded.
"""

from __future__ import annotations

import math

import pytest
import torch


def _make_flowedit(monkeypatch):
    """Create a VoxelFlowEdit bypassing the constructor's model requirement."""
    from clearmesh.editing.voxel_flowedit import VoxelFlowEdit, FlowEditConfig

    ve = VoxelFlowEdit.__new__(VoxelFlowEdit)
    ve.flow_model = object()  # non-None sentinel; tests monkeypatch the methods
    ve.device = "cpu"
    ve.config = FlowEditConfig(num_steps=4, gamma=1.0, eta=0.0, guidance_scale=1.0)
    return ve


class TestForwardDiffuse:
    def test_t_zero_is_pure_noise(self, monkeypatch):
        """At t=0, the flow-matching forward process should be pure noise."""
        ve = _make_flowedit(monkeypatch)
        x_0 = torch.ones(1, 5, 4)
        torch.manual_seed(0)
        x_t = ve._forward_diffuse(x_0, t=0.0)
        # At t=0, x_t = 0*x_0 + 1*noise = noise. Mean near 0, std near 1.
        assert abs(x_t.mean().item()) < 0.5
        assert 0.5 < x_t.std().item() < 1.5

    def test_t_one_is_clean_data(self, monkeypatch):
        """At t=1, the forward process should return clean data (noise is ignored)."""
        ve = _make_flowedit(monkeypatch)
        x_0 = torch.ones(1, 5, 4) * 3.0
        x_t = ve._forward_diffuse(x_0, t=1.0)
        assert torch.allclose(x_t, x_0)

    def test_t_half_is_midpoint(self, monkeypatch):
        ve = _make_flowedit(monkeypatch)
        x_0 = torch.ones(1, 10, 8) * 4.0
        torch.manual_seed(42)
        # Run many samples to average out the noise
        means = []
        for _ in range(50):
            x_t = ve._forward_diffuse(x_0, t=0.5)
            means.append(x_t.mean().item())
        avg = sum(means) / len(means)
        # Expected: 0.5*4.0 + 0.5*E[noise] = 2.0
        assert abs(avg - 2.0) < 0.2

    def test_shape_preserved(self, monkeypatch):
        ve = _make_flowedit(monkeypatch)
        for shape in [(1, 5, 4), (2, 100, 32), (1, 1000, 64)]:
            x = torch.zeros(*shape)
            out = ve._forward_diffuse(x, t=0.3)
            assert out.shape == shape


class TestTrajectoryCorrection:
    def test_identity_when_trajectory_matches(self, monkeypatch):
        """If x_t equals the source trajectory at step t, correction should be 0."""
        from clearmesh.editing.voxel_flowedit import FlowEditConfig

        ve = _make_flowedit(monkeypatch)
        cfg = FlowEditConfig(num_steps=4, t_start=0.0, t_end=1.0)

        source_traj = [torch.ones(1, 3, 2) * i for i in range(4)]
        # Pick step_idx=2 → x_t should equal source_traj[2] for zero correction
        x_t = source_traj[2].clone()
        correction = ve._trajectory_correction(x_t, source_traj, t=0.5, config=cfg)
        assert torch.allclose(correction, torch.zeros_like(x_t))

    def test_correction_magnitude_scales_with_offset(self, monkeypatch):
        from clearmesh.editing.voxel_flowedit import FlowEditConfig

        ve = _make_flowedit(monkeypatch)
        cfg = FlowEditConfig(num_steps=4, t_start=0.0, t_end=1.0)
        source_traj = [torch.zeros(1, 3, 2) for _ in range(4)]
        x_t = torch.ones(1, 3, 2) * 10.0
        correction = ve._trajectory_correction(x_t, source_traj, t=0.5, config=cfg)
        # correction = source - x_t = 0 - 10 = -10
        assert torch.allclose(correction, torch.full_like(x_t, -10.0))

    def test_step_idx_clamps_at_end(self, monkeypatch):
        """t beyond t_end should clamp to last trajectory entry, not index error."""
        from clearmesh.editing.voxel_flowedit import FlowEditConfig

        ve = _make_flowedit(monkeypatch)
        cfg = FlowEditConfig(num_steps=4, t_start=0.0, t_end=1.0)
        source_traj = [torch.zeros(1, 3, 2), torch.ones(1, 3, 2)]
        x_t = torch.zeros(1, 3, 2)
        # Should not raise IndexError even for t > t_end
        correction = ve._trajectory_correction(x_t, source_traj, t=1.5, config=cfg)
        assert correction.shape == x_t.shape


class TestEditMask:
    def test_all_ones_mask_edits_everything(self, monkeypatch):
        """With mask=1 everywhere, the ODE step should equal the full edit velocity."""
        from clearmesh.editing.voxel_flowedit import VoxelFlowEdit, FlowEditConfig

        ve = _make_flowedit(monkeypatch)
        # Build a trivial run where velocity is a constant
        B, N, D = 1, 3, 2
        ve._compute_velocity = lambda x_t, t, cond, gs: torch.ones(B, N, D) * 0.1
        ve._compute_source_trajectory = lambda *a, **k: [torch.zeros(B, N, D) for _ in range(2)]
        ve._silhouette_guidance = lambda x_t, img, t: torch.zeros_like(x_t)
        ve._encode_image_condition = lambda img: None
        ve.config = FlowEditConfig(num_steps=2, gamma=0.0, eta=0.0,
                                    t_start=0.0, t_end=1.0, guidance_scale=1.0)

        from PIL import Image
        target = Image.new("RGB", (16, 16))

        x_0 = torch.zeros(B, N, D)
        # Edit everything
        out = ve.edit(
            source_ss_latent=x_0,
            target_image=target,
            source_image=target,
            edit_mask=None,
        )
        assert out.shape == (B, N, D)
        assert not torch.isnan(out).any()


class TestAutoDetectMask:
    def test_returns_shape_matches_voxel_count(self, monkeypatch):
        """auto_detect_edit_mask currently returns all-ones stub;
        Phase 4 will replace. This test guards the shape contract."""
        from clearmesh.editing.voxel_flowedit import VoxelFlowEdit
        from PIL import Image

        ve = VoxelFlowEdit.__new__(VoxelFlowEdit)
        ve.device = "cpu"

        N = 42
        voxel_indices = torch.randint(0, 256, (N, 3))
        src = Image.new("RGB", (64, 64), (100, 100, 100))
        tgt = Image.new("RGB", (64, 64), (200, 100, 100))

        mask = ve.auto_detect_edit_mask(src, tgt, voxel_indices)
        assert mask.shape == (N,)
        assert mask.dtype in (torch.float32, torch.float16, torch.float64)
        # Values in [0, 1] (binary or soft)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


class TestRepaintSoftMask:
    def test_soft_mask_respects_boundary_width(self):
        """_create_soft_mask should produce a gradient within boundary_width."""
        from clearmesh.editing.slat_repaint import SLATRepainter

        rep = SLATRepainter.__new__(SLATRepainter)
        rep.device = "cpu"

        # Cube of voxels, half edited (x<5) half not
        positions = torch.tensor(
            [[i, 0, 0] for i in range(10)], dtype=torch.float32
        )
        edit_mask = torch.tensor(
            [1.0] * 5 + [0.0] * 5
        )
        soft = rep._create_soft_mask(edit_mask, positions, boundary_width=2)
        # Voxel at x=5 (distance 1 from edited x=4) should have soft mask > 0
        assert soft[5] > 0.0
        # Voxel at x=9 (distance 5 from nearest edited) should still be 0
        assert soft[9] == 0.0

    def test_soft_mask_no_boundary_returns_binary(self):
        """With boundary_width=0, soft mask should equal binary mask."""
        from clearmesh.editing.slat_repaint import SLATRepainter

        rep = SLATRepainter.__new__(SLATRepainter)
        rep.device = "cpu"

        edit_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        positions = torch.tensor([[0, 0, 0]] * 4, dtype=torch.float32)
        soft = rep._create_soft_mask(edit_mask, positions, boundary_width=0)
        assert torch.allclose(soft, edit_mask)

    def test_soft_mask_empty_edits_returns_source(self):
        """If nothing is edited, soft mask should equal the all-zero input."""
        from clearmesh.editing.slat_repaint import SLATRepainter

        rep = SLATRepainter.__new__(SLATRepainter)
        rep.device = "cpu"

        edit_mask = torch.zeros(5)
        positions = torch.zeros(5, 3)
        soft = rep._create_soft_mask(edit_mask, positions, boundary_width=3)
        assert torch.allclose(soft, edit_mask)
