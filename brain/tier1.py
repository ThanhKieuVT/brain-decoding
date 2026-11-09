"""Đánh giá tầng 1 (mức embedding): metric theo từng ảnh cho mỗi điều kiện, và PRR.

Đầu vào là embedding dự đoán cho từng điều kiện (real, shuf-x, shuf-w, null-0, ...),
embedding đích của ảnh thật, và nhãn category. Đầu ra là bảng dài theo cột của PLAN.md:
``image_id, condition, k, metric, value, seed``, cộng một bảng tóm tắt PRR.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common.metrics import prr_with_ci, topk_retrieval, two_way_identification

__all__ = ["Condition", "METRICS", "per_item_scores", "evaluate", "summarize_prr"]


@dataclass
class Condition:
    """Embedding dự đoán của một điều kiện. ``items`` là chỉ số ảnh hợp lệ (mặc định: tất cả)."""

    name: str
    pred: np.ndarray
    items: np.ndarray | None = None
    k: int | None = None
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _chance_topk(n: int, k: int) -> float:
    return k / n


# tên metric -> (hàm(pred, target, groups) trả điểm theo ảnh, hàm(n) trả mức ngẫu nhiên)
METRICS = {
    "twoway_pearson": (lambda p, t, g: two_way_identification(p, t, "pearson"), lambda n: 0.5),
    "twoway_cosine": (lambda p, t, g: two_way_identification(p, t, "cosine"), lambda n: 0.5),
    "twoway_within_cat": (
        lambda p, t, g: two_way_identification(p, t, "cosine", groups=g),
        lambda n: 0.5,
    ),
    "top1": (lambda p, t, g: topk_retrieval(p, t, 1), lambda n: _chance_topk(n, 1)),
    "top5": (lambda p, t, g: topk_retrieval(p, t, 5), lambda n: _chance_topk(n, 5)),
}


def per_item_scores(
    pred: np.ndarray, target: np.ndarray, groups: np.ndarray, metrics: list[str]
) -> dict[str, np.ndarray]:
    """Điểm theo từng ảnh. Mọi metric tính trên cùng tập ứng viên là toàn bộ ``target``."""
    return {m: METRICS[m][0](pred, target, groups) for m in metrics}


def evaluate(
    conditions: list[Condition],
    target: np.ndarray,
    groups: np.ndarray,
    image_ids: list[str],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Bảng dài các điểm theo ảnh.

    Với điều kiện có ``items`` (ví dụ shuf-w loại ảnh không có bạn cùng category), dự đoán
    của các ảnh bị loại được thay bằng dự đoán thật để giữ nguyên tập ứng viên, rồi chỉ báo
    cáo điểm của các ảnh hợp lệ.
    """
    metrics = metrics or list(METRICS)
    real = next((c for c in conditions if c.name == "real"), None)
    rows = []
    for c in conditions:
        pred = c.pred
        items = np.arange(target.shape[0]) if c.items is None else np.asarray(c.items)
        if pred.shape[0] != target.shape[0]:
            if real is None:
                raise ValueError(f"{c.name}: thiếu điều kiện 'real' để lấp các ảnh bị loại")
            full = real.pred.copy()
            full[items] = pred
            pred = full
        scores = per_item_scores(pred, target, groups, metrics)
        for m, v in scores.items():
            for i in items:
                rows.append((image_ids[i], c.name, c.k, m, float(v[i]), c.seed))
    return pd.DataFrame(rows, columns=["image_id", "condition", "k", "metric", "value", "seed"])


def summarize_prr(
    df: pd.DataFrame, n_items: int, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """PRR của mỗi (điều kiện, k, metric) so với 'real', ghép cặp theo ảnh.

    Với điều kiện có nhiều seed (nhiều hoán vị), điểm theo ảnh được lấy trung bình qua
    seed trước khi tính PRR.
    """
    out = []
    agg = df.groupby(["condition", "k", "metric", "image_id"], dropna=False)["value"].mean()
    agg = agg.reset_index()
    real = agg[agg.condition == "real"].set_index(["metric", "image_id"])["value"]
    for (cond, k, metric), g in agg.groupby(["condition", "k", "metric"], dropna=False):
        chance = METRICS[metric][1](n_items)
        r = real.loc[metric].reindex(g.image_id).to_numpy()
        c = g.value.to_numpy()
        est, lo, hi = prr_with_ci(c, r, chance, n_boot=n_boot, seed=seed)
        out.append(
            {
                "condition": cond,
                "k": k,
                "metric": metric,
                "n_images": int(np.isfinite(c).sum()),
                "mean_control": float(np.nanmean(c)),
                "mean_real_same_images": float(np.nanmean(r)),
                "chance": chance,
                "prr": est,
                "prr_lo": lo,
                "prr_hi": hi,
            }
        )
    return pd.DataFrame(out)
