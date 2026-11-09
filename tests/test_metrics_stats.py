import numpy as np
import pytest

from common.metrics import (
    prr,
    prr_with_ci,
    similarity_matrix,
    topk_retrieval,
    two_way_identification,
)
from common.stats import bootstrap_ci, bootstrap_ratio_of_means, holm, paired_wilcoxon


def test_two_way_hand_example():
    # 3 ảnh, embedding 3 chiều. pred_0 đúng hướng target_0; pred_1 giống target_2 hơn.
    target = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    pred = np.array([[1.0, 0, 0], [0, 0.2, 1], [0, 0, 1]])
    s = two_way_identification(pred, target, "cosine")
    # ảnh 0: sim(t0,p0)=1 > sim(t0,p1)=0, sim(t0,p2)=0 -> 1
    # ảnh 1: sim(t1,p1)=0.196 > sim(t1,p0)=0, sim(t1,p2)=0 -> 1
    # ảnh 2: sim(t2,p2)=1 > sim(t2,p0)=0, nhưng < sim(t2,p1)? sim(t2,p1)=0.98 < 1 -> 1
    np.testing.assert_allclose(s, [1, 1, 1])
    pred_bad = np.array([[0, 1.0, 0], [1, 0, 0], [0, 0, 1]])
    s2 = two_way_identification(pred_bad, target, "cosine")
    np.testing.assert_allclose(s2, [0.25, 0.25, 1.0])  # hoà ở j còn lại tính 0,5


def test_two_way_random_near_chance_and_perfect():
    rng = np.random.default_rng(0)
    t = rng.standard_normal((200, 32))
    assert np.allclose(two_way_identification(t, t), 1.0)
    r = two_way_identification(rng.standard_normal((200, 32)), t)
    assert abs(r.mean() - 0.5) < 0.05


def test_two_way_groups_restrict_and_nan():
    rng = np.random.default_rng(1)
    t = rng.standard_normal((6, 8))
    g = np.array([0, 0, 1, 1, 2, -1])
    s = two_way_identification(t, t, "cosine", groups=g)
    assert np.isnan(s[4]) and np.isnan(s[5])  # nhóm một ảnh và ảnh không nhãn
    np.testing.assert_allclose(s[:4], 1.0)


def test_similarity_pearson_matches_numpy():
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal((5, 10)), rng.standard_normal((5, 10))
    c = similarity_matrix(a, b, "pearson")
    assert np.isclose(c[1, 3], np.corrcoef(a[1], b[3])[0, 1])


def test_topk():
    t = np.eye(4)
    p = np.array([[1.0, 0, 0, 0], [0.9, 0.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_array_equal(topk_retrieval(p, t, 1), [1, 0, 1, 1])
    np.testing.assert_array_equal(topk_retrieval(p, t, 2), [1, 1, 1, 1])


def test_prr_values():
    assert prr(0.5, 0.9, 0.5) == 0.0
    assert prr(0.9, 0.9, 0.5) == 1.0
    assert prr(0.7, 0.9, 0.5) == pytest.approx(0.5)
    assert np.isnan(prr(0.6, 0.5, 0.5))


def test_prr_with_ci_ratio_of_means():
    rng = np.random.default_rng(3)
    real = 0.5 + 0.4 * rng.random(300)
    ctrl = 0.5 + 0.2 * rng.random(300)
    est, lo, hi = prr_with_ci(ctrl, real, 0.5, n_boot=500)
    assert est == pytest.approx((ctrl.mean() - 0.5) / (real.mean() - 0.5))
    assert lo < est < hi


def test_stats_helpers():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    est, lo, hi = bootstrap_ci(np.arange(100.0), n_boot=300)
    assert lo < est < hi and est == pytest.approx(49.5)
    r, _, _ = bootstrap_ratio_of_means(np.ones(10), 2 * np.ones(10), n_boot=50)
    assert r == pytest.approx(0.5)
    _, p = paired_wilcoxon(np.arange(20.0), np.arange(20.0) + 1)
    assert p < 0.001
