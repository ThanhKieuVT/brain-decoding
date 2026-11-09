"""Đọc, chuyển đổi và lấy lát cắt dữ liệu THINGS-EEG2 đã tiền xử lý (63 kênh, 250 Hz).

File gốc trên Hugging Face (``LidongYang/EEG_Image_decode``, thư mục
``Preprocessed_data_250Hz/sub-XX/``) là dict lưu bằng pickle, kiểu float64, gồm
``preprocessed_eeg_data`` (ảnh × lần lặp × kênh × thời gian), ``ch_names`` và ``times``.
Hàm ``convert_subject`` ghi lại thành ``.npy`` thường để mở bằng memmap, tiết kiệm RAM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "load_pickled_eeg",
    "convert_subject",
    "EEGSplit",
    "open_split",
    "average_repeats",
    "concept_id_from_folder",
    "list_image_conditions",
    "find_condition_dirs",
]

SPLITS = {"training": "preprocessed_eeg_training.npy", "test": "preprocessed_eeg_test.npy"}


def load_pickled_eeg(path: str | Path) -> dict:
    """Đọc file gốc. Chấp nhận cả dict pickle lẫn mảng object 0 chiều chứa dict."""
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict) or "preprocessed_eeg_data" not in obj:
        raise ValueError(f"{path}: không phải định dạng THINGS-EEG2 mong đợi")
    return obj


def convert_subject(
    src_dir: str | Path,
    dst_dir: str | Path,
    dtype: str = "float32",
    chunk: int = 512,
    check_rows: int = 64,
) -> dict:
    """Chuyển hai file gốc của một subject sang ``{split}_eeg.npy`` và ``{split}_meta.json``.

    Ghi theo từng khúc vào memmap trên đĩa, nên RAM đỉnh xấp xỉ kích thước một file gốc.
    Ghi lại sai số tuyệt đối lớn nhất do đổi kiểu, đo trên ``check_rows`` ảnh đầu tiên.
    """
    src, dst = Path(src_dir), Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, fname in SPLITS.items():
        raw = load_pickled_eeg(src / fname)
        data = raw["preprocessed_eeg_data"]
        out = np.lib.format.open_memmap(
            dst / f"{split}_eeg.npy", mode="w+", dtype=dtype, shape=data.shape
        )
        for s in range(0, data.shape[0], chunk):
            out[s : s + chunk] = data[s : s + chunk].astype(dtype)
        out.flush()
        sample = data[:check_rows]
        err = float(np.abs(sample - out[:check_rows].astype(np.float64)).max())
        scale = float(np.abs(sample).max())
        meta = {
            "split": split,
            "shape": list(data.shape),
            "axes": ["image", "repeat", "channel", "time"],
            "dtype": dtype,
            "source_dtype": str(data.dtype),
            "ch_names": [str(c) for c in raw["ch_names"]],
            "times": np.asarray(raw["times"], dtype=np.float64).round(6).tolist(),
            "max_abs_conversion_error_first_rows": err,
            "max_abs_value_first_rows": scale,
            "source_file": fname,
        }
        (dst / f"{split}_meta.json").write_text(json.dumps(meta, indent=1))
        report[split] = {k: meta[k] for k in ("shape", "dtype", "source_dtype")}
        report[split]["max_abs_err"] = err
        report[split]["max_abs_value"] = scale
        del raw, data, out
    return report


@dataclass
class EEGSplit:
    """Dữ liệu một split: ``data`` có trục ảnh × lần lặp × kênh × thời gian."""

    data: np.ndarray
    ch_names: list[str]
    times: np.ndarray

    def time_slice(self, t_start: float, t_end: float) -> EEGSplit:
        """Giữ các mẫu thời gian trong [t_start, t_end) tính bằng giây."""
        keep = (self.times >= t_start - 1e-9) & (self.times < t_end - 1e-9)
        return EEGSplit(self.data[..., keep], self.ch_names, self.times[keep])


def open_split(subject_dir: str | Path, split: str, mmap: bool = True) -> EEGSplit:
    """Mở dữ liệu đã chuyển đổi; ``mmap=True`` không nạp cả mảng vào RAM."""
    d = Path(subject_dir)
    meta = json.loads((d / f"{split}_meta.json").read_text())
    data = np.load(d / f"{split}_eeg.npy", mmap_mode="r" if mmap else None)
    if list(data.shape) != meta["shape"]:
        raise ValueError(f"{d}: kích thước {data.shape} khác meta {meta['shape']}")
    return EEGSplit(data, meta["ch_names"], np.asarray(meta["times"]))


def average_repeats(
    data: np.ndarray, n: int | None = None, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Trung bình theo trục lần lặp. ``n`` lần lặp chọn ngẫu nhiên (cho đường cong H3)."""
    if n is None or n >= data.shape[1]:
        return np.asarray(data, dtype=np.float32).mean(axis=1)
    if rng is None:
        raise ValueError("Cần rng khi chọn ngẫu nhiên n lần lặp")
    idx = np.stack([rng.choice(data.shape[1], size=n, replace=False) for _ in range(data.shape[0])])
    picked = np.take_along_axis(np.asarray(data), idx[:, :, None, None], axis=1)
    return picked.astype(np.float32).mean(axis=1)


_FOLDER = re.compile(r"^\d+_(.+)$")


def concept_id_from_folder(name: str) -> str:
    """'00042_aircraft_carrier' -> 'aircraft_carrier' (khớp cột uniqueID của THINGS)."""
    m = _FOLDER.match(name)
    if not m:
        raise ValueError(f"Tên thư mục không đúng dạng số_tên: {name}")
    return m.group(1)


def list_image_conditions(image_dir: str | Path) -> list[str]:
    """Danh sách thư mục concept theo thứ tự tên, đúng thứ tự ảnh trong file EEG."""
    return sorted(p.name for p in Path(image_dir).iterdir() if p.is_dir())


def find_condition_dirs(
    root: str | Path, expected: tuple[int, ...] = (1654, 200)
) -> dict[int, Path]:
    """Tìm thư mục chứa đúng ``expected`` thư mục con dạng ``số_tên`` (ảnh train, ảnh test).

    Dùng sau khi giải nén gói ảnh, vì cấu trúc thư mục bên trong gói có thể khác nhau.
    """
    found: dict[int, Path] = {}
    for d in [Path(root), *Path(root).rglob("*")]:
        if not d.is_dir():
            continue
        n = sum(1 for c in d.iterdir() if c.is_dir() and _FOLDER.match(c.name))
        if n in expected and n not in found:
            found[n] = d
    missing = [n for n in expected if n not in found]
    if missing:
        raise FileNotFoundError(f"Không thấy thư mục có {missing} concept dưới {root}")
    return found
