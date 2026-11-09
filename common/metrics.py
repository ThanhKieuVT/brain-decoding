"""Metric ở mức embedding và chỉ số Prior Reliance Ratio (PRR).

Quy ước: ``target`` là embedding của ảnh thật, ``pred`` là embedding decoder dự đoán,
cùng thứ tự theo ảnh. Các hàm trả về điểm theo từng ảnh để bootstrap theo ảnh.
"""

from __future__ import annotations

import numpy as np

from common.stats import bootstrap_ratio_of_means

__all__ = [
    "similarity_matrix",
    "two_way_identification",
    "topk_retrieval",
    "prr",
    "prr_with_ci",
]


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


def similarity_matrix(target: np.ndarray, pred: np.ndarray, metric: str = "pearson") -> np.ndarray:
    """Ma trận C[i, j] = sim(target_i, pred_j).

    ``metric="pearson"`` theo giao thức MindEye2 cho metric ảnh;
    ``metric="cosine"`` dùng cho retrieval trong không gian embedding.
    """
    if target.shape != pred.shape:
        raise ValueError(f"target {target.shape} và pred {pred.shape} phải cùng kích thước")
    if metric == "pearson":
        a, b = _zscore_rows(target), _zscore_rows(pred)
    elif metric == "cosine":
        a, b = _unit_rows(target), _unit_rows(pred)
    else:
        raise ValueError(f"metric không hỗ trợ: {metric}")
    return a @ b.T


def two_way_identification(
    pred: np.ndarray,
    target: np.ndarray,
    metric: str = "pearson",
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Two-way identification theo từng ảnh; mức ngẫu nhiên là 0,5.

    Với ảnh i, so ``sim(target_i, pred_i)`` với ``sim(target_i, pred_j)`` cho mọi j khác i,
    rồi lấy tỉ lệ thắng (hoà tính 0,5). Nếu có ``groups``, chỉ so với j cùng nhóm
    (two-way với ảnh nhiễu cùng category); ảnh không có bạn cùng nhóm, hoặc có nhãn âm,
    nhận NaN.
    """
    c = similarity_matrix(target, pred, metric)
    n = c.shape[0]
    diag = np.diag(c)[:, None]
    wins = (diag > c).astype(np.float64) + 0.5 * (diag == c)
    mask = ~np.eye(n, dtype=bool)
    if groups is not None:
        g = np.asarray(groups)
        if g.shape != (n,):
            raise ValueError("groups phải có một nhãn cho mỗi ảnh")
        mask &= g[:, None] == g[None, :]
        mask &= (g >= 0)[:, None]
    counts = mask.sum(axis=1)
    scores = np.where(mask, wins, 0.0).sum(axis=1)
    out = np.full(n, np.nan)
    ok = counts > 0
    out[ok] = scores[ok] / counts[ok]
    return out


def topk_retrieval(
    pred: np.ndarray, candidates: np.ndarray, k: int = 1, metric: str = "cosine"
) -> np.ndarray:
    """Retrieval top-k theo từng ảnh: ứng viên đúng của ảnh i là ``candidates[i]``.

    Trả về mảng 0/1. Mức ngẫu nhiên là k / số ứng viên.
    """
    c = similarity_matrix(candidates, pred, metric).T  # c[i, j] = sim(pred_i, cand_j)
    correct = np.diag(c)[:, None]
    rank = (c > correct).sum(axis=1)  # số ứng viên tốt hơn đáp án đúng
    return (rank < k).astype(np.float64)


def prr(m_control: float, m_real: float, m_chance: float) -> float:
    """PRR_c(M) = (M_c - M_chance) / (M_real - M_chance).

    Gần 0: điểm đến từ tín hiệu đúng. Gần 1: prior hoặc category gánh gần hết.
    Trả về NaN khi điểm thật không cao hơn mức ngẫu nhiên.
    """
    denom = m_real - m_chance
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    return float((m_control - m_chance) / denom)


def prr_with_ci(
    control_items: np.ndarray,
    real_items: np.ndarray,
    m_chance: float,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """PRR là tỉ số của hai giá trị trung bình, kèm khoảng tin cậy bootstrap theo ảnh.

    Không lấy trung bình các tỉ số theo từng ảnh. Bootstrap có cặp: cùng một bộ ảnh
    được lấy lại cho điều kiện control và điều kiện thật.
    """
    c = np.asarray(control_items, dtype=np.float64) - m_chance
    r = np.asarray(real_items, dtype=np.float64) - m_chance
    return bootstrap_ratio_of_means(c, r, n_boot=n_boot, seed=seed, alpha=alpha)
