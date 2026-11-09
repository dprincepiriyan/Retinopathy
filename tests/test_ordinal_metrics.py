"""Unit tests for Ordinal Losses, Soft-QWK, GeM Pooling, ResViT-Pro, and Nelder-Mead Optimizer."""

import numpy as np
import torch

from src.losses.asymmetric_focal_loss import AsymmetricFocalLoss
from src.losses.ordinal_focal_loss import (
    CORALOrdinalFocalLoss,
    coral_logits_to_probs,
    encode_ordinal_targets,
)
from src.losses.soft_qwk_loss import SoftQWKLoss
from src.tier3_grading.gem_pooling import GeMPooling2d
from src.tier3_grading.resvit_pro import ResViTPro
from src.tier4_ensemble.qwk_threshold_optimizer import NelderMeadQWKOptimizer
from src.utils.metrics import quadratic_weighted_kappa


def test_encode_ordinal_targets_and_coral_loss() -> None:
    labels = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    encoded = encode_ordinal_targets(labels, num_classes=5)
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert torch.equal(encoded, expected)

    loss_fn = CORALOrdinalFocalLoss(num_classes=5)
    # Logits matching ground truth should yield much lower loss than inverted logits
    good_logits = (expected * 6.0) - 3.0
    bad_logits = -good_logits
    assert loss_fn(good_logits, labels).item() < loss_fn(bad_logits, labels).item()

    probs = coral_logits_to_probs(good_logits)
    assert probs.shape == (5, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-4)
    assert torch.equal(torch.argmax(probs, dim=1), labels)


def test_soft_qwk_loss_penalizes_distant_grades_quadratically() -> None:
    loss_fn = SoftQWKLoss(num_classes=5)
    targets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    # Perfect prediction probabilities
    perfect_probs = torch.eye(5)
    # Off-by-1 prediction probabilities
    near_logits = torch.tensor(
        [
            [2.0, 3.0, -3.0, -3.0, -3.0],
            [2.0, 2.0, 3.0, -3.0, -3.0],
            [-3.0, 2.0, 2.0, 3.0, -3.0],
            [-3.0, -3.0, 2.0, 2.0, 3.0],
            [-3.0, -3.0, -3.0, 3.0, 2.0],
        ]
    )
    # Extreme opposite prediction probabilities (predicting 4 for 0 and 0 for 4)
    far_logits = torch.flip(perfect_probs * 6.0, dims=[1])

    l_perfect = loss_fn(perfect_probs, targets, is_probs=True)
    l_near = loss_fn(near_logits, targets, is_probs=False)
    l_far = loss_fn(far_logits, targets, is_probs=False)

    assert l_perfect.item() < 0.05
    assert l_perfect.item() < l_near.item() < l_far.item()


def test_gem_pooling_amplifies_focal_microlesion_spike() -> None:
    # Create a 64x64 feature map that is near zero everywhere except a single 2x2 microaneurysm peak
    feat = torch.full((1, 4, 64, 64), 0.01)
    feat[:, :, 30:32, 30:32] = 5.0

    gap = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
    gem = GeMPooling2d(initial_p=4.0, learnable=True)(feat)

    # GeM pooling should produce a substantially higher activation than Global Average Pooling
    assert torch.all(gem > gap * 4.0)


def test_resvit_pro_forward_and_rank_consistent_biases() -> None:
    model = ResViTPro(fpn_dim=32, num_classes=5, num_heads=4).eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    tri = out["tri_head"]
    assert tri.ordinal_logits.shape == (2, 4)
    assert tri.class_probs.shape == (2, 5)
    assert tri.referable_prob.shape == (2, 1)
    assert tri.continuous_score.shape == (2, 1)
    assert out["lesion_maps"].shape[1] == 4

    # Verify rank-consistency: P(y > 0) >= P(y > 1) >= P(y > 2) >= P(y > 3)
    cum_probs = torch.sigmoid(tri.ordinal_logits)
    diffs = cum_probs[:, :-1] - cum_probs[:, 1:]
    assert torch.all(diffs >= -1e-6)


def test_nelder_mead_qwk_optimizer_improves_shifted_thresholds() -> None:
    rng = np.random.default_rng(123)
    y_true = rng.choice([0, 1, 2, 3, 4], size=400, p=[0.50, 0.15, 0.15, 0.10, 0.10])
    # Simulate systematically shifted continuous scores (e.g., compressed by 0.8x + 0.25 offset)
    cont_scores = y_true * 0.8 + 0.25 + rng.normal(0.0, 0.18, size=400)

    default_preds = NelderMeadQWKOptimizer.discretize(cont_scores, [0.5, 1.5, 2.5, 3.5])
    default_qwk = quadratic_weighted_kappa(y_true, default_preds)

    opt = NelderMeadQWKOptimizer(initial_thresholds=[0.5, 1.5, 2.5, 3.5]).fit(cont_scores, y_true)
    opt_preds = opt.predict(cont_scores)
    opt_qwk = quadratic_weighted_kappa(y_true, opt_preds)

    assert opt_qwk >= default_qwk
    assert opt_qwk > 0.85
    assert np.all(np.diff(opt.thresholds_) > 0)


def test_asymmetric_focal_loss_multi_label() -> None:
    asl = AsymmetricFocalLoss(gamma_neg=4.0, gamma_pos=1.0)
    logits = torch.tensor([[3.0, -3.0, 2.5], [-2.5, 3.0, -3.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    loss = asl(logits, targets)
    loss.backward()
    assert loss.item() > 0.0
    assert logits.grad is not None
