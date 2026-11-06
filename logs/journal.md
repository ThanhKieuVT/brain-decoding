# Nhật ký làm việc

## 2026-11-06 – T00: dựng repo và môi trường
- Đã làm: tạo cấu trúc mục 4, `.gitignore`, env conda `braindec` (Python 3.11, phiên bản ghim trong `requirements.txt`), `logs/hardware.md`, workflow CI, test khói.
- Kết quả: `pytest -q` đạt 8/8, ruff và black sạch. Workflow CI chưa chạy thật vì repo chưa có remote.
- Sai khác so với PLAN: `.gitignore` dùng `/data/` thay vì `data/`, vì `data/` sẽ chặn cả code trong `brain/data/` và `gaze/data/`.
- Vấn đề còn lại: máy là Apple M4 16 GB, không có CUDA, ổ còn 22 GiB (dưới ngưỡng 50 GB của mục 10). T02 và T03 cần chị quyết định trước khi tải dữ liệu.
- Vấn đề còn lại: chưa có `docs/Plan_nghien_cuu_ban2.pdf` và `.claude/settings.json` (mục 6.3, chị tự thêm). Tên `scripts/queue.py` sẽ che module `queue` của Python khi chạy trực tiếp, nên cần đổi tên khi tới task đó.
