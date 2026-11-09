"""Các control thay tín hiệu não thật bằng tín hiệu đã bị làm hỏng.

Mọi hàm shuffle trả về mảng ``donor`` cùng độ dài số ảnh: ảnh i nhận tín hiệu của ảnh
``donor[i]``. Giá trị -1 nghĩa là ảnh i không có control hợp lệ và phải bị loại khỏi
phép tính. Vì encoder xử lý từng trial độc lập, xáo tín hiệu đầu vào rồi mã hoá cho cùng
kết quả với mã hoá rồi xáo embedding; điều này cho phép chạy shuffle ở mức embedding.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "shuffle_within_category",
    "shuffle_cross_category",
    "apply_donor",
    "null_zero",
    "null_gauss",
    "PCABottleneck",
    "nearest_train",
]


def shuffle_within_category(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """shuf-w: mỗi ảnh nhận tín hiệu của một ảnh khác **cùng category**, không bao giờ là chính nó.

    Trong mỗi nhóm có ít nhất 2 ảnh, xếp ngẫu nhiên các ảnh thành vòng rồi lấy ảnh kế tiếp,
    nên không ảnh nào tự ghép với mình. Ảnh có nhãn âm (không có category, hoặc bị loại vì
    nhiều category) và ảnh là thành viên duy nhất của nhóm nhận -1.
    """
    g = np.asarray(groups)
    donor = np.full(g.shape[0], -1, dtype=np.int64)
    for label in np.unique(g[g >= 0]):
        members = np.flatnonzero(g == label)
        if members.size < 2:
            continue
        order = rng.permutation(members)
        donor[order] = np.roll(order, -1)
    return donor


def shuffle_cross_category(
    groups: np.ndarray, rng: np.random.Generator, max_iter: int = 10_000
) -> np.ndarray:
    """shuf-x: hoán vị trong đó mỗi ảnh nhận tín hiệu của một ảnh **khác category**.

    Ảnh có nhãn âm được coi là một category riêng của chính nó, nên có thể ghép với
    bất kỳ ảnh nào khác. Dùng hoán vị (không lặp donor) để giữ nguyên phân phối tín hiệu.
    """
    g = np.asarray(groups).copy()
    n = g.shape[0]
    neg = g < 0
    g[neg] = g.max(initial=0) + 1 + np.arange(neg.sum())  # nhãn riêng cho từng ảnh không nhãn
    perm = rng.permutation(n)
    for _ in range(max_iter):
        bad = np.flatnonzero(g[perm] == g)
        if bad.size == 0:
            return perm
        for i in bad:
            j = rng.integers(n)
            if g[perm[j]] != g[i] and g[perm[i]] != g[j]:
                perm[i], perm[j] = perm[j], perm[i]
    raise RuntimeError("Không tìm được hoán vị khác category; một category chiếm quá nửa số ảnh?")


def apply_donor(x: np.ndarray, donor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (tín hiệu đã xáo cho các ảnh hợp lệ, chỉ số các ảnh hợp lệ)."""
    keep = np.flatnonzero(donor >= 0)
    return x[donor[keep]], keep


def null_zero(x: np.ndarray) -> np.ndarray:
    """null-0: toàn số 0, cùng kích thước với tín hiệu thật."""
    return np.zeros_like(x)


def null_gauss(
    shape: tuple[int, ...], mean: np.ndarray, std: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """null-g: nhiễu Gauss độc lập, khớp trung bình và độ lệch chuẩn **tính trên train**.

    ``mean`` và ``std`` có kích thước của một mẫu (ví dụ kênh × thời gian).
    """
    noise = rng.standard_normal(shape).astype(np.float32)
    return noise * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)


class PCABottleneck:
    """pca-k: chiếu tín hiệu qua PCA hạng k học trên train, rồi dựng lại trong không gian gốc.

    Kết quả có cùng kích thước với đầu vào, nên đưa thẳng vào decoder như tín hiệu thật.
    """

    def __init__(self, max_k: int, seed: int = 0) -> None:
        self.max_k = max_k
        self.seed = seed
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, train: np.ndarray) -> PCABottleneck:
        from sklearn.utils.extmath import randomized_svd

        flat = np.asarray(train, dtype=np.float32).reshape(train.shape[0], -1)
        self.mean_ = flat.mean(axis=0)
        centered = flat - self.mean_
        _, s, vt = randomized_svd(centered, n_components=self.max_k, random_state=self.seed)
        self.components_ = vt.astype(np.float32)
        total = float((centered.astype(np.float64) ** 2).sum())
        self.explained_variance_ratio_ = (s.astype(np.float64) ** 2) / total
        return self

    def transform(self, x: np.ndarray, k: int) -> np.ndarray:
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("Gọi fit() trên tập train trước")
        if not 1 <= k <= self.max_k:
            raise ValueError(f"k phải nằm trong [1, {self.max_k}]")
        shape = x.shape
        flat = np.asarray(x, dtype=np.float32).reshape(shape[0], -1) - self.mean_
        v = self.components_[:k]
        return ((flat @ v.T) @ v + self.mean_).reshape(shape)


def nearest_train(pred: np.ndarray, train_emb: np.ndarray) -> np.ndarray:
    """nn-train: chỉ số ảnh train có embedding gần dự đoán nhất (cosine)."""
    p = pred / np.linalg.norm(pred, axis=1, keepdims=True)
    t = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True)
    return np.argmax(p @ t.T, axis=1)
