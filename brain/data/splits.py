"""Tách validation từ tập train theo concept, để chọn checkpoint không dùng tập test."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

__all__ = ["ConceptSplit", "concept_val_split"]


@dataclass
class ConceptSplit:
    seed: int
    n_concepts: int
    images_per_concept: int
    val_concepts: list[int]
    train_concepts: list[int]

    def image_indices(self, which: str) -> np.ndarray:
        """Chỉ số ảnh (hàng trong file EEG train) của 'train' hoặc 'val'."""
        concepts = self.val_concepts if which == "val" else self.train_concepts
        k = self.images_per_concept
        return (np.asarray(concepts)[:, None] * k + np.arange(k)[None, :]).ravel()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: str | Path) -> ConceptSplit:
        return cls(**json.loads(Path(path).read_text()))


def concept_val_split(
    n_concepts: int = 1654, n_val: int = 74, images_per_concept: int = 10, seed: int = 0
) -> ConceptSplit:
    """Chọn ngẫu nhiên ``n_val`` concept làm validation, không trùng concept với train.

    Mặc định 74 concept × 10 ảnh = 740 ảnh, cùng cỡ với validation của NICE, nhưng tách
    theo concept để mô phỏng đúng bài toán zero-shot của tập test. File EEG train xếp ảnh
    liền nhau theo concept, mỗi concept ``images_per_concept`` ảnh.
    """
    rng = np.random.default_rng(seed)
    val = np.sort(rng.choice(n_concepts, size=n_val, replace=False))
    train = np.setdiff1d(np.arange(n_concepts), val)
    return ConceptSplit(seed, n_concepts, images_per_concept, val.tolist(), train.tolist())
