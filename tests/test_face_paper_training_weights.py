from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "research" / "train_face_paper_faithful.py"
    spec = importlib.util.spec_from_file_location("train_face_paper_faithful", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample(module):
    return module.PaperFaceSample(
        path=Path("sample.npz"),
        point_features=np.asarray(
            [
                [-1.0, -1.0, -1.0, 0.0, 0.0, 1.0],
                [1.0, -1.0, -1.0, 0.0, 1.0, 0.0],
                [-1.0, 1.0, -1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, -1.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        ),
        tokens=np.asarray(
            [
                [0, 0, 0, 0, 0, 7, 0, 7, 0],
                [0, 0, 7, 0, 7, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        ),
        paper_within_face_order="rotate_min_zyx",
    )


def test_first_face_and_topology_training_weights_are_opt_in():
    torch = pytest.importorskip("torch")
    module = _load_module()
    sample = _sample(module)

    baseline = module._make_batch(
        [sample],
        max_faces=4,
        device=torch.device("cpu"),
        num_bins=8,
    )
    weighted = module._make_batch(
        [sample],
        max_faces=4,
        device=torch.device("cpu"),
        num_bins=8,
        first_face_loss_weight=3.0,
        topology_reuse_weight=0.5,
        topology_edge_closure_weight=1.0,
    )

    baseline_weights = baseline[3]
    weighted_weights = weighted[3]

    assert float(baseline_weights[0, 0].mean()) == pytest.approx(1.0)
    assert float(baseline_weights[0, 1].mean()) == pytest.approx(1.0)
    assert float(weighted_weights[0, 0].mean()) == pytest.approx(3.0)
    assert float(weighted_weights[0, 1].max()) > 1.0
    assert float(weighted_weights[0, 2:].sum()) == pytest.approx(0.0)


def test_loss_face_prefix_count_masks_coordinate_loss_without_changing_inputs():
    torch = pytest.importorskip("torch")
    module = _load_module()
    sample = _sample(module)

    batch = module._make_batch(
        [sample],
        max_faces=4,
        device=torch.device("cpu"),
        num_bins=8,
        first_face_loss_weight=5.0,
        loss_face_prefix_count=1,
    )

    input_faces, target_faces, valid_weights, eos_targets, eos_weights = batch[1], batch[2], batch[3], batch[4], batch[5]

    assert target_faces[0, 1, 0].item() == 0
    assert float(valid_weights[0, 0].mean()) == pytest.approx(5.0)
    assert float(valid_weights[0, 1:].sum()) == pytest.approx(0.0)
    assert float(eos_weights[0, :2].sum()) == pytest.approx(2.0)
    assert float(eos_targets[0, 1]) == pytest.approx(1.0)
    assert input_faces.shape == (1, 4, 9)


def test_input_face_token_noise_is_opt_in_and_only_changes_decoder_inputs():
    torch = pytest.importorskip("torch")
    module = _load_module()
    sample = _sample(module)
    sample = module.PaperFaceSample(
        path=sample.path,
        point_features=sample.point_features,
        tokens=np.asarray(
            [
                [3, 3, 3, 3, 3, 4, 3, 4, 3],
                [3, 3, 4, 3, 4, 3, 4, 3, 3],
                [4, 4, 4, 4, 4, 5, 4, 5, 4],
            ],
            dtype=np.int64,
        ),
        paper_within_face_order=sample.paper_within_face_order,
    )

    baseline = module._make_batch(
        [sample],
        max_faces=5,
        device=torch.device("cpu"),
        num_bins=8,
    )
    torch.manual_seed(7)
    noisy = module._make_batch(
        [sample],
        max_faces=5,
        device=torch.device("cpu"),
        num_bins=8,
        input_face_token_noise_prob=1.0,
        input_face_token_noise_max_offset=1,
        input_face_noise_prefix_count=1,
    )

    baseline_inputs, baseline_targets = baseline[1], baseline[2]
    noisy_inputs, noisy_targets = noisy[1], noisy[2]

    assert torch.equal(noisy_targets, baseline_targets)
    assert torch.equal(noisy_inputs[:, 0], baseline_inputs[:, 0])
    assert not torch.equal(noisy_inputs[:, 1], baseline_inputs[:, 1])
    assert torch.equal(noisy_inputs[:, 2:], baseline_inputs[:, 2:])


def test_first_face_tie_marginal_matches_ce_for_singleton_group():
    torch = pytest.importorskip("torch")
    module = _load_module()
    num_bins = 8
    target_faces = torch.tensor([[[0, 0, 0, 0, 0, 4, 0, 4, 0]]], dtype=torch.long)
    valid_weights = torch.ones((1, 1, 9), dtype=torch.float32)
    hidden = torch.zeros((1, 1, 4), dtype=torch.float32)
    logits = torch.zeros((1, 1, 9, num_bins), dtype=torch.float32)

    class UniformModel:
        def _causal_logits_from_hidden(self, hidden_arg, target_arg):
            return torch.zeros((target_arg.shape[0], target_arg.shape[1], 9, num_bins), dtype=torch.float32)

    baseline_loss, baseline_weight = module._compute_loss(
        torch.nn.functional,
        logits,
        target_faces,
        valid_weights,
        num_bins,
    )
    marginal_loss, marginal_weight = module._compute_loss_with_first_face_tie_marginal(
        torch.nn.functional,
        UniformModel(),
        hidden,
        logits,
        target_faces,
        valid_weights,
        num_bins,
    )

    assert float(marginal_weight) == pytest.approx(float(baseline_weight))
    assert float(marginal_loss) == pytest.approx(float(baseline_loss), rel=1e-6, abs=1e-6)


def test_first_face_tie_marginal_credits_same_min_non_row0_face():
    torch = pytest.importorskip("torch")
    module = _load_module()
    num_bins = 8
    target_faces = torch.tensor(
        [
            [
                [0, 0, 0, 0, 0, 4, 0, 4, 0],
                [0, 0, 0, 0, 4, 0, 4, 0, 0],
                [3, 3, 3, 3, 4, 3, 4, 3, 3],
            ]
        ],
        dtype=torch.long,
    )
    valid_weights = torch.zeros((1, 3, 9), dtype=torch.float32)
    valid_weights[:, 0, :] = 1.0
    hidden = torch.zeros((1, 3, 4), dtype=torch.float32)
    logits = torch.zeros((1, 3, 9, num_bins), dtype=torch.float32)
    # Ordinary CE is deliberately bad for row 0.
    logits[:, 0, :, 7] = 8.0

    class CandidateAwareModel:
        def _causal_logits_from_hidden(self, hidden_arg, target_arg):
            out = torch.full((target_arg.shape[0], target_arg.shape[1], 9, num_bins), -8.0, dtype=torch.float32)
            for row in range(target_arg.shape[0]):
                for slot in range(9):
                    out[row, 0, slot, int(target_arg[row, 0, slot])] = 8.0
            return out

    baseline_loss, _ = module._compute_loss(
        torch.nn.functional,
        logits,
        target_faces,
        valid_weights,
        num_bins,
    )
    marginal_loss, marginal_weight = module._compute_loss_with_first_face_tie_marginal(
        torch.nn.functional,
        CandidateAwareModel(),
        hidden,
        logits,
        target_faces,
        valid_weights,
        num_bins,
    )

    assert float(marginal_weight) == pytest.approx(9.0)
    assert float(marginal_loss) < float(baseline_loss) * 0.1
