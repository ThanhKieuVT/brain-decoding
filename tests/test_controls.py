import numpy as np

from brain.controls import (
    PCABottleneck,
    apply_donor,
    nearest_train,
    null_gauss,
    null_zero,
    shuffle_cross_category,
    shuffle_within_category,
)


def _groups():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 8, size=200)
    g[:5] = -1  # không có category
    g[5] = 99  # nhóm chỉ một ảnh
    return g


def test_within_category_never_self_same_group():
    g = _groups()
    for seed in range(20):
        d = shuffle_within_category(g, np.random.default_rng(seed))
        ok = d >= 0
        assert not np.any(d[ok] == np.flatnonzero(ok))
        assert np.all(g[d[ok]] == g[ok])
        assert np.all(d[:6] == -1)  # ảnh không nhãn và ảnh lẻ bị loại
        assert len(set(d[ok].tolist())) == ok.sum()  # mỗi ảnh làm donor đúng một lần


def test_cross_category_is_permutation_with_different_groups():
    g = _groups()
    for seed in range(20):
        p = shuffle_cross_category(g, np.random.default_rng(seed))
        assert sorted(p.tolist()) == list(range(len(g)))
        assert not np.any(p == np.arange(len(g)))
        lab = g[g >= 0]
        both = (g >= 0) & (g[p] >= 0)
        assert np.all(g[p][both] != g[both]) and lab.size > 0


def test_apply_donor_and_nulls():
    x = np.arange(12.0).reshape(4, 3)
    d = np.array([1, -1, 3, 0])
    xs, keep = apply_donor(x, d)
    np.testing.assert_array_equal(keep, [0, 2, 3])
    np.testing.assert_array_equal(xs, x[[1, 3, 0]])
    assert np.all(null_zero(x) == 0)
    z = null_gauss(
        (5000, 2, 3), np.full((2, 3), 2.0), np.full((2, 3), 0.5), np.random.default_rng(0)
    )
    assert abs(z.mean() - 2.0) < 0.02 and abs(z.std() - 0.5) < 0.02


def test_pca_bottleneck_reconstructs_low_rank():
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((3, 4, 5))
    train = np.einsum("nk,kct->nct", rng.standard_normal((300, 3)), basis) + 1.0
    test = np.einsum("nk,kct->nct", rng.standard_normal((10, 3)), basis) + 1.0
    pca = PCABottleneck(max_k=5).fit(train)
    np.testing.assert_allclose(pca.transform(test, 3), test, atol=1e-3)
    err1 = np.abs(pca.transform(test, 1) - test).mean()
    assert err1 > 0.1  # hạng 1 mất thông tin
    assert pca.explained_variance_ratio_[:3].sum() > 0.999


def test_nearest_train():
    train = np.eye(5)
    pred = np.array([[0.1, 0.9, 0, 0, 0], [0, 0, 0, 0.2, 1]])
    np.testing.assert_array_equal(nearest_train(pred, train), [1, 4])
