# Cấu hình máy

Ghi ngày 2026-11-06 (T00).

| Mục | Giá trị |
|---|---|
| Máy | Apple M4, arm64, macOS 15.1 (24B2083) |
| CPU | 10 nhân (4 hiệu năng, 6 tiết kiệm điện) |
| GPU | Apple M4 tích hợp, 10 nhân, Metal 3 |
| VRAM | Không có VRAM riêng, GPU dùng chung RAM 16 GB |
| Driver NVIDIA | Không có (`nvidia-smi` không tồn tại) |
| CUDA | Không có (`nvcc` không tồn tại, `torch.cuda.is_available()` = False) |
| Tăng tốc PyTorch | MPS khả dụng, đã chạy thử phép nhân ma trận thành công |
| RAM | 16 GB |
| Ổ đĩa | 460 GiB, còn trống 22 GiB sau khi cài môi trường (25 GiB trước khi cài) |

## Môi trường

- Conda env `braindec`, Python 3.11.18, PyTorch 2.15.0, chiếm 1,3 GB.
- Chạy bằng `/opt/anaconda3/envs/braindec/bin/python`. Hàm `conda` trong shell không tương tác báo lỗi vì `CONDA_EXE` rỗng, nên gọi thẳng `/opt/anaconda3/bin/conda`.

## Hệ quả với PLAN

- Ổ trống dưới ngưỡng 50 GB của mục 10, nên chưa được tải dữ liệu.
- Không có CUDA, nên `torch.cuda.max_memory_allocated` ở T01 không đo được trên máy này.
