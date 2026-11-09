"""Bootstrap theo ảnh, kiểm định Wilcoxon có cặp và hiệu chỉnh Holm."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats as _st

__all__ = ["bootstrap_ci", "bootstrap_ratio_of_means", "paired_wilcoxon", "holm"]


def _valid(*arrays: np.ndarray) -> np.ndarray:
    ok = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        ok &= np.isfinite(a)
    return ok


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    stat: Callable[[np.ndarray], float] = np.mean,
) -> tuple[float, float, float]:
    """Ước lượng và khoảng percentile (1 - alpha), lấy lại mẫu theo ảnh. Bỏ NaN."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.array([stat(v[i]) for i in idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi)


def bootstrap_ratio_of_means(
    num: np.ndarray,
    den: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """mean(num) / mean(den) với bootstrap có cặp theo ảnh. Mẫu bootstrap có mẫu số <= 0 bị bỏ."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    ok = _valid(num, den)
    num, den = num[ok], den[ok]
    if num.size == 0 or den.mean() <= 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, num.size, size=(n_boot, num.size))
    d = den[idx].mean(axis=1)
    r = num[idx].mean(axis=1)
    good = d > 0
    boots = r[good] / d[good]
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(num.mean() / den.mean()), float(lo), float(hi)


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank có cặp, hai phía. Trả về (thống kê, p). Bỏ cặp có NaN."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = _valid(a, b)
    res = _st.wilcoxon(a[ok], b[ok])
    return float(res.statistic), float(res.pvalue)


def holm(pvals: Sequence[float]) -> np.ndarray:
    """P-value đã hiệu chỉnh Holm–Bonferroni, giữ thứ tự đầu vào."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
