"""Đo tài nguyên cho một khối code: giây/mẫu, VRAM đỉnh, RAM đỉnh, dung lượng thư mục.

Dùng trong code:
    with Profiler("pilot_eeg", decoder="atm", modality="eeg", subject="sub-01",
                  n_items=200, stage="retrieval") as prof:
        ...
Kết quả được nối vào ``results/pilot/profile.csv``.

Chạy thử: ``python -m bench.profile --task selftest``
"""

from __future__ import annotations

import argparse
import csv
import os
import threading
import time
from pathlib import Path

import psutil

COLUMNS = [
    "task", "decoder", "modality", "subject", "n_items", "stage", "sec_per_item",
    "peak_vram_gb", "peak_ram_gb", "disk_gb", "fp16", "batch_size", "notes",
]  # fmt: skip
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "results" / "pilot" / "profile.csv"


def dir_size_gb(path: str | Path | None) -> float:
    if not path or not Path(path).exists():
        return float("nan")
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


class _RamSampler(threading.Thread):
    def __init__(self, interval: float = 0.05) -> None:
        super().__init__(daemon=True)
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = self.proc.memory_info().rss
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def stop(self) -> int:
        self._halt.set()
        self.join()
        return max(self.peak, self.proc.memory_info().rss)


class Profiler:
    """Context manager đo một khối code và ghi một dòng vào CSV."""

    def __init__(
        self,
        task: str,
        n_items: int,
        decoder: str = "",
        modality: str = "",
        subject: str = "",
        stage: str = "",
        data_dir: str | Path | None = None,
        fp16: bool = False,
        batch_size: int | None = None,
        notes: str = "",
        csv_path: str | Path = DEFAULT_CSV,
    ) -> None:
        self.row = {
            "task": task, "decoder": decoder, "modality": modality, "subject": subject,
            "n_items": n_items, "stage": stage, "fp16": fp16,
            "batch_size": batch_size if batch_size is not None else "", "notes": notes,
        }  # fmt: skip
        self.data_dir = data_dir
        self.csv_path = Path(csv_path)

    def __enter__(self) -> Profiler:
        self._torch = None
        try:
            import torch

            self._torch = torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        self._ram = _RamSampler()
        self._ram.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        torch = self._torch
        vram = float("nan")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            vram = torch.cuda.max_memory_allocated() / 1e9
        elif torch is not None and torch.backends.mps.is_available():
            vram = torch.mps.driver_allocated_memory() / 1e9  # MPS không có số đỉnh
        elapsed = time.perf_counter() - self._t0
        peak_ram = self._ram.stop() / 1e9
        n = max(int(self.row["n_items"]), 1)
        self.row.update(
            sec_per_item=round(elapsed / n, 6),
            peak_vram_gb=round(vram, 3),
            peak_ram_gb=round(peak_ram, 3),
            disk_gb=round(dir_size_gb(self.data_dir), 3),
        )
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerow(self.row)


def _selftest(csv_path: Path, device: str) -> None:
    import torch

    model = torch.nn.Sequential(
        torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, 64)
    )
    model.to(device)
    x = torch.randn(1024, 512, device=device)
    with Profiler("selftest", n_items=1024, decoder="dummy_mlp", stage="forward",
                  notes=f"device={device}", csv_path=csv_path):  # fmt: skip
        with torch.no_grad():
            for _ in range(20):
                model(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="selftest", choices=["selftest"])
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()
    _selftest(Path(args.csv), args.device)
    print(f"Đã ghi một dòng vào {args.csv}")


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
