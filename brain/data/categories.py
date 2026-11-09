"""Gán 27 category cấp cao của THINGS cho các concept của THINGS-EEG2.

Nguồn (OSF dự án jum2f, thư mục 03_category-level): ``category27_manual.tsv`` có 1854 hàng
theo đúng thứ tự concept trong ``concepts-metadata_things.tsv`` và 27 cột nhị phân.
Một concept có thể thuộc 0, 1 hoặc nhiều category.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

__all__ = ["load_category27", "assign_categories", "NO_CATEGORY", "MULTI_CATEGORY"]

NO_CATEGORY = -1
MULTI_CATEGORY = -2


def load_category27(
    category_tsv: str | Path, concepts_tsv: str | Path
) -> tuple[list[str], dict[str, list[str]]]:
    """Trả về (danh sách 27 category, dict uniqueID -> các category của concept đó)."""
    with open(category_tsv, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    names, cat_rows = rows[0], rows[1:]
    with open(concepts_tsv, encoding="utf-8") as f:
        meta = list(csv.DictReader(f, delimiter="\t"))
    if len(meta) != len(cat_rows):
        raise ValueError(f"Số concept lệch: {len(meta)} metadata, {len(cat_rows)} category")
    mapping = {}
    for m, r in zip(meta, cat_rows, strict=True):
        mapping[m["uniqueID"]] = [n for n, v in zip(names, r, strict=True) if int(float(v)) == 1]
    return names, mapping


def assign_categories(
    concept_ids: list[str],
    names: list[str],
    mapping: dict[str, list[str]],
    multi: str = "exclude",
) -> tuple[np.ndarray, dict]:
    """Nhãn số nguyên cho từng concept.

    ``multi="exclude"``: concept thuộc nhiều category nhận ``MULTI_CATEGORY`` (bị loại khỏi
    shuf-w). ``multi="first"``: lấy category đầu tiên theo thứ tự cột. Concept không có
    category nhận ``NO_CATEGORY``. Concept không tìm thấy trong THINGS gây lỗi.
    """
    missing = [c for c in concept_ids if c not in mapping]
    if missing:
        raise KeyError(f"{len(missing)} concept không có trong THINGS, ví dụ {missing[:5]}")
    index = {n: i for i, n in enumerate(names)}
    labels = np.empty(len(concept_ids), dtype=np.int64)
    for i, c in enumerate(concept_ids):
        cats = mapping[c]
        if not cats:
            labels[i] = NO_CATEGORY
        elif len(cats) == 1 or multi == "first":
            labels[i] = index[cats[0]]
        elif multi == "exclude":
            labels[i] = MULTI_CATEGORY
        else:
            raise ValueError(f"multi không hỗ trợ: {multi}")
    counts = Counter(labels.tolist())
    summary = {
        "n_concepts": len(concept_ids),
        "no_category": counts.pop(NO_CATEGORY, 0),
        "multi_category": counts.pop(MULTI_CATEGORY, 0),
        "per_category": {names[k]: v for k, v in sorted(counts.items())},
        "usable_for_shuf_w": int(sum(v for v in counts.values() if v >= 2)),
    }
    return labels, summary
