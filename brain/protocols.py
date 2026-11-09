"""Giao thức dữ liệu và đánh giá của từng bài gốc, chép đúng theo code đã đọc.

Nguồn (nhánh ``main``, đọc ngày 06/11/2026):
- NICE: github.com/eeyhsong/NICE-EEG @ a7709d8, file ``nice_stand.py``.
- ATM: github.com/dongyangli-del/EEG_Image_decode @ 49cb3a2, file
  ``Retrieval/ATMS_retrieval.py`` và ``Retrieval/eegdatasets_leaveone.py``.

Không bài nào cố định seed: NICE rút seed bằng ``np.random.randint(2023)`` mà không seed
trước, ATM không đặt seed. Vì vậy không thể tái tạo đúng từng con số; chỉ có thể tái tạo
đúng quy trình và báo cáo độ dao động qua nhiều seed.

Hai điểm cố ý khác bài gốc, ghi rõ để báo cáo:
1. ATM nhánh ``main`` chọn epoch theo độ chính xác test. Quy tắc 1.4 của PLAN.md cấm việc
   này, nên dùng epoch cuối và trung bình 10 epoch cuối (Bảng 4 của bài ATM cũng báo cáo
   trung bình 10 epoch cuối).
2. Seed được cố định và ghi lại.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Protocol",
    "NICE",
    "ATM",
    "AUDIT",
    "PROTOCOLS",
    "prepare_eeg",
    "nice_val_split",
    "atm_time_mask",
    "kway_top1",
]


@dataclass(frozen=True)
class Protocol:
    name: str
    train_repeats: str  # "mean": trung bình 4 lần lặp; "separate": mỗi lần lặp là một mẫu
    test_repeats: str  # "mean": trung bình 80 lần lặp
    time_rule: str  # "all": giữ mọi mẫu; "atm": times[50:] rồi giữ 0 <= t <= 1.0
    val_rule: str  # "nice_image_740", "none_test_selected", "concept_74"
    test_target: str  # đích retrieval của 200 ảnh test
    clip_model: str
    epochs: int
    batch_size: int
    lr: float
    optimizer: str
    selection: str  # cách chọn checkpoint mà dự án dùng
    paper_selection: str  # cách chọn checkpoint trong code gốc
    notes: str = ""


NICE = Protocol(
    name="nice",
    train_repeats="mean",
    test_repeats="mean",
    time_rule="all",
    val_rule="nice_image_740",
    test_target="concept_center",  # dnn_feature/center_clip.npy, (200, 768)
    clip_model="CLIP ViT-L/14",
    epochs=200,
    batch_size=1000,
    lr=2e-4,
    optimizer="Adam(betas=(0.5, 0.999))",
    selection="min_val_loss",
    paper_selection="min_val_loss",
    notes="Validation: xáo 16.540 ảnh (sau khi trung bình lần lặp), lấy 740 ảnh đầu.",
)

ATM = Protocol(
    name="atm",
    train_repeats="separate",
    test_repeats="mean",
    time_rule="atm",
    val_rule="none_test_selected",
    test_target="viewed_image",  # ViT-H-14_features_test.pt, img_features
    clip_model="OpenCLIP ViT-H-14 laion2b_s32b_b79k",
    epochs=40,
    batch_size=64,
    lr=3e-4,
    optimizer="AdamW",
    selection="last_epoch_and_mean_last10",
    paper_selection="best_test_accuracy",
    notes=(
        "Loss = 0.99 * ClipLoss(ảnh) + 0.01 * ClipLoss(text). "
        "Test k-way: 2, 4, 10, 50, 100, 200."
    ),
)

AUDIT = Protocol(
    name="audit",
    train_repeats="mean",
    test_repeats="mean",
    time_rule="all",
    val_rule="concept_74",
    test_target="viewed_image",
    clip_model="theo decoder",
    epochs=0,
    batch_size=0,
    lr=0.0,
    optimizer="theo decoder",
    selection="min_val_loss",
    paper_selection="",
    notes="Giao thức chung để so PRR giữa các decoder; không dùng để so với số bài gốc.",
)

PROTOCOLS = {p.name: p for p in (NICE, ATM, AUDIT)}


def atm_time_mask(times: np.ndarray, n_samples: int, start: float = 0.0, end: float = 1.0):
    """Mặt nạ thời gian đúng như ATM: ``t = times[50:]``, giữ ``start <= t <= end``.

    ATM áp mặt nạ có độ dài ``len(times) - 50`` lên trục thời gian của dữ liệu. Phép này chỉ
    hợp lệ khi độ dài khớp; nếu không khớp thì báo lỗi để kiểm tra lại dữ liệu thật,
    thay vì đoán.
    """
    t = np.asarray(times)[50:]
    if t.shape[0] != n_samples:
        raise ValueError(
            f"ATM cắt times[50:] còn {t.shape[0]} giá trị, nhưng dữ liệu có {n_samples} mẫu "
            "thời gian. Cần kiểm tra file thật trước khi tái hiện ATM."
        )
    return (t >= start) & (t <= end)


def prepare_eeg(
    data: np.ndarray, times: np.ndarray, protocol: Protocol, part: str
) -> tuple[np.ndarray, np.ndarray]:
    """Chuẩn bị EEG theo giao thức. ``data``: ảnh × lần lặp × kênh × thời gian.

    Trả về (mẫu, chỉ số ảnh của từng mẫu). Kiểu float32, như ``.float()`` trong code gốc.
    """
    rule = protocol.train_repeats if part == "train" else protocol.test_repeats
    x = np.asarray(data, dtype=np.float32)
    if protocol.time_rule == "atm":
        x = x[..., atm_time_mask(times, x.shape[-1])]
    elif protocol.time_rule != "all":
        raise ValueError(protocol.time_rule)
    n_img, n_rep = x.shape[:2]
    if rule == "mean":
        return x.mean(axis=1), np.arange(n_img)
    if rule == "separate":
        return x.reshape(n_img * n_rep, *x.shape[2:]), np.repeat(np.arange(n_img), n_rep)
    raise ValueError(rule)


def nice_val_split(n_images: int = 16540, n_val: int = 740, seed: int = 0):
    """Validation kiểu NICE: hoán vị ngẫu nhiên các ảnh train, 740 ảnh đầu là validation.

    Tách theo ảnh, nên validation và train có chung concept. NICE không cố định seed;
    ở đây seed được cố định và phải được ghi lại cùng kết quả.
    """
    perm = np.random.default_rng(seed).permutation(n_images)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def kway_top1(
    pred: np.ndarray, candidates: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Độ chính xác k-way như ``evaluate_model`` của ATM, theo từng ảnh (0/1).

    Với ảnh i: chọn ngẫu nhiên k-1 lớp khác i, thêm lớp i, đúng nếu lớp i có điểm cao nhất
    (tích vô hướng, như ``logit_scale * eeg @ img.T``). Với k bằng số ứng viên thì không
    còn ngẫu nhiên. Mức ngẫu nhiên là 1/k.
    """
    s = np.asarray(pred, dtype=np.float64) @ np.asarray(candidates, dtype=np.float64).T
    n = s.shape[1]
    out = np.empty(s.shape[0])
    for i in range(s.shape[0]):
        others = np.delete(np.arange(n), i)
        chosen = others if k >= n else rng.choice(others, size=k - 1, replace=False)
        out[i] = float(s[i, i] > s[i, chosen].max())
    return out
