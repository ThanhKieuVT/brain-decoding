# CLAUDE.md

Repo nghiên cứu của chị Kiều: Project 1 (Brain Decoding, chính) và Project 3 (Gaze Guidance, có điều kiện).
Kế hoạch chi tiết và danh sách task: **đọc `PLAN.md` trước khi làm bất cứ việc gì.**

## Quy tắc không được vi phạm
1. **Không đọc, mở, in, xem hay tóm tắt** bất kỳ file nào trong `data/physionet/` (DUA của PhysioNet cấm đưa dữ liệu cho dịch vụ LLM online). Code Project 3 chỉ viết và test trên dữ liệu giả (`gaze/synthetic.py`). Khi chạy trên dữ liệu thật, chỉ được in số tổng hợp.
2. **Hỏi chị trước khi** tải thứ gì > 5 GB hoặc chạy job dự kiến > 2 giờ. Báo dung lượng/thời gian ước tính.
3. **Không dùng ảnh test để chỉnh tham số.** Ảnh pilot chỉ dùng để đo tốc độ.
4. **Không commit** gì trong `data/`, checkpoint, ảnh sinh ra.
5. Không sửa số liệu bằng tay; kết quả bất ngờ thì báo nguyên trạng.

## Mỗi phiên làm việc
- Đọc mục 3 (Trạng thái) trong `PLAN.md`, làm task `TODO` có số nhỏ nhất (trừ khi chị chỉ định).
- Xong task: cập nhật bảng trạng thái, ghi 3–5 dòng vào `logs/journal.md`, commit với mã task (ví dụ `T11: control trên embedding`).
- Gặp quyết định thiết kế: dừng, đưa 2–3 phương án, hỏi chị.

## Lệnh hay dùng
- Test: `pytest -q`
- Đo tài nguyên: `python -m bench.profile --task <tên>`
- Tạo lại toàn bộ bảng/hình: `bash scripts/make_all.sh`
