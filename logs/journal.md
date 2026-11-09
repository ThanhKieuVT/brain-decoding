# Nhật ký làm việc

## 2026-11-06 – T00: dựng repo và môi trường
- Đã làm: tạo cấu trúc mục 4, `.gitignore`, env conda `braindec` (Python 3.11, phiên bản ghim trong `requirements.txt`), `logs/hardware.md`, workflow CI, test khói.
- Kết quả: `pytest -q` đạt 8/8, ruff và black sạch. Workflow CI chưa chạy thật vì repo chưa có remote.
- Sai khác so với PLAN: `.gitignore` dùng `/data/` thay vì `data/`, vì `data/` sẽ chặn cả code trong `brain/data/` và `gaze/data/`.
- Vấn đề còn lại: máy là Apple M4 16 GB, không có CUDA, ổ còn 22 GiB (dưới ngưỡng 50 GB của mục 10). T02 và T03 cần chị quyết định trước khi tải dữ liệu.
- Vấn đề còn lại: chưa có `docs/Plan_nghien_cuu_ban2.pdf` và `.claude/settings.json` (mục 6.3, chị tự thêm). Tên `scripts/queue.py` sẽ che module `queue` của Python khi chạy trực tiếp, nên cần đổi tên khi tới task đó.

## 2026-11-09 – T01, T10 (một phần), T11: code nền nhánh EEG và notebook Colab
- Đã làm: loader THINGS-EEG2 (đọc pickle, chuyển float16 theo khúc, memmap), tách validation theo concept, gán 27 category THINGS, sáu control, metric embedding và PRR có bootstrap, `bench/profile.py`, hai notebook Colab, `docs/colab_data_plan.md`.
- Kết quả: `pytest -q` đạt 29/29, ruff và black sạch. Notebook mới được kiểm tra cú pháp, chưa chạy trên Colab.
- Số đo: bản 63 kênh 250 Hz trên Hugging Face nặng 10,35 GB mỗi subject (float64); float16 còn khoảng 2,59 GB. Đọc pickle protocol 4 tốn RAM khoảng 1,1 lần kích thước mảng (đo trên máy).
- Vấn đề còn lại: trong 1.854 concept THINGS, 559 không có category và 302 có nhiều category (bảng manual), cần chị chốt cách xử lý cho shuf-w. Chưa có wrapper train NICE và ATM, chưa có metric ảnh.

## 2026-11-09 – T12 (chuẩn bị): ghi lại giao thức NICE và ATM
- Đã làm: đọc code `main` của NICE (a7709d8) và ATM (49cb3a2), ghi giao thức vào `brain/protocols.py`: cửa sổ thời gian, lần lặp, validation, đích retrieval, k-way. Đổi lưu trữ mặc định sang float32 vì code gốc đưa float32 vào model.
- Kết quả: `pytest -q` đạt 34/34.
- Phát hiện: NICE rút seed ngẫu nhiên không ghi lại, ATM không đặt seed, nên chỉ tái tạo được quy trình, không tái tạo được đúng con số. ATM cắt `times[50:]` làm mặt nạ cho trục thời gian 250 mẫu; cần kiểm tra trên file thật.
- Vấn đề còn lại: float32 cho 10 subject cần khoảng 52 GB trên Drive.
