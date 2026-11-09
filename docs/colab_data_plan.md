# Đề xuất dữ liệu và cách chạy Project 1 (nhánh EEG) trên Google Colab

Cập nhật: 09/11/2026. Phạm vi: tuần 0–2 của kế hoạch ICCV, nhánh EEG. Nhánh fMRI (NSD) chưa có trong tài liệu này vì NSD chưa được duyệt.

## 1. Chọn bản dữ liệu

Dùng bản THINGS-EEG2 đã tiền xử lý **63 kênh, 250 Hz** trên Hugging Face (`LidongYang/EEG_Image_decode`, thư mục `Preprocessed_data_250Hz/`). Đây là bản NICE và ATM dùng trong bài báo, nên số tái hiện so được với số gốc. Bản 17 kênh, 100 Hz của tác giả bộ dữ liệu (OSF, dự án 3JK45) cho kết quả không so trực tiếp được với NICE và ATM. Việc chốt bản nào vẫn là một mục của tuần 0; code hiện chỉ hỗ trợ bản 63 kênh.

Mỗi file EEG là một dict lưu bằng pickle, kiểu float64, gồm `preprocessed_eeg_data` (ảnh × lần lặp × kênh × thời gian), `ch_names` và `times`.

## 2. Tải gì, không tải gì

Dung lượng lấy từ API của Hugging Face ngày 09/11/2026.

| File | Dung lượng | Tải? | Lý do |
|---|---|---|---|
| `Preprocessed_data_250Hz/sub-XX/preprocessed_eeg_training.npy` | 8,34 GB mỗi subject | Có, từng subject một | Dữ liệu train: 16.540 ảnh × 4 lần lặp × 63 kênh × 250 mẫu |
| `Preprocessed_data_250Hz/sub-XX/preprocessed_eeg_test.npy` | 2,02 GB mỗi subject | Có | Dữ liệu test: 200 ảnh × 80 lần lặp; cần giữ từng lần lặp cho đường cong H3 |
| `images_set.tar.gz` | 0,78 GB | Có | Ảnh kích thích, dùng tên thư mục để ghép concept và để sinh embedding CLIP |
| `ViT-H-14_features_train.pt`, `ViT-H-14_features_test.pt` | khoảng 0,07 GB | Có | Embedding CLIP của ảnh mà ATM dùng làm đích |
| `emb_eeg/ATM_S_eeg_features_sub-XX_test.pt` | dưới 1 MB | Có, để kiểm tra pipeline | Embedding EEG do ATM xuất; checkpoint của ATM chọn theo test nên không dùng làm kết quả |
| `train_image_latent_512.pt` | 1,09 GB | Để sau | Chỉ cần khi chạy nhánh sinh ảnh mức thấp của ATM |
| `generated_imgs.tar.gz` | 8,13 GB | Không | Ảnh ATM đã sinh sẵn; dự án tự sinh theo giao thức của mình |
| `preprocessed_MEG/` | khoảng 43 GB | Không | Không thuộc phạm vi |
| Metadata THINGS: `category27_*.tsv`, `concepts-metadata_things.tsv` (OSF jum2f) | dưới 1 MB | Có | Gán 27 category cấp cao |

Tổng tải xuống nếu lấy đủ 10 subject là khoảng 104 GB. Notebook tải từng subject vào ổ tạm của Colab, chuyển đổi, rồi xoá bản gốc, nên ổ tạm chỉ cần chứa một subject mỗi lúc.

## 3. Lưu trữ trên Google Drive

Bản gốc float64 quá lớn để giữ trên Drive. Notebook chuyển sang **float16** và lưu dạng `.npy` thường:

| | Mỗi subject | 10 subject |
|---|---|---|
| float64 gốc | 10,35 GB | 103,5 GB |
| float16 | khoảng 2,59 GB | khoảng 25,9 GB |
| float32, nếu cần | khoảng 5,18 GB | khoảng 51,8 GB |

Drive miễn phí có 15 GB, không đủ cho 10 subject kể cả ở float16. Có ba cách:

1. **Mua gói Google One 100 GB (em đề xuất).** Đủ cho cả 10 subject ở float16 cùng ảnh, embedding và kết quả.
2. Giữ trên Drive miễn phí khoảng 4–5 subject một lúc, chạy xong thì thay subject.
3. Dùng float32 nếu kiểm tra cho thấy float16 làm sai kết quả. Cách này cần gói 100 GB.

Notebook ghi lại sai số lớn nhất khi đổi kiểu, đo trên 64 ảnh đầu, vào `training_meta.json` và `test_meta.json`. Chị xem con số này ở lần chạy sub-01 trước khi chạy tiếp các subject khác.

Cấu trúc thư mục trên Drive:

```
MyDrive/BrainDecoding/
  code/                         file zip code (nếu không dùng GitHub)
  data/
    things_eeg2/
      eeg_63ch_250hz_f16/sub-01/  training_eeg.npy, training_meta.json, test_eeg.npy, test_meta.json
      archives/                    images_set.tar.gz, ViT-H-14_features_*.pt
      val_split_seed0.json         concept dùng làm validation
    things_meta/                   category27_*.tsv, concepts-metadata_things.tsv
  results/                       CSV kết quả, profile.csv
  logs/                          cấu hình máy và pip freeze của từng phiên
```

Drive chậm khi đọc hàng chục nghìn file nhỏ, nên gói ảnh được giữ nguyên dạng tar trên Drive và giải nén vào `/content` mỗi phiên.

## 4. Runtime Colab

- **Chuyển đổi dữ liệu:** không cần GPU. Đo trên máy cho thấy đọc một dict pickle (protocol 4 trở lên) tốn RAM khoảng 1,1 lần kích thước mảng. File train float64 là 8,34 GB, nên cần khoảng 9–10 GB RAM. Runtime thường của Colab có khoảng 12–13 GB, vừa đủ nhưng sát. Nếu bị tràn RAM, chọn High-RAM.
- **Train NICE, ATM và sinh ảnh:** chưa đo. Mọi bước đều được bọc trong `bench.profile.Profiler` và ghi vào `results/pilot/profile.csv` trên Drive, để có số thật trước khi ước lượng thời gian cho 10 subject × 3 seed.
- Không cài lại torch trên Colab. `requirements-colab.txt` chỉ cài các gói còn thiếu; phiên bản thực tế được ghi vào `logs/pip_freeze_*.txt`.

## 5. Các bước chạy

1. Trên máy: commit code, rồi `bash scripts/pack_for_colab.sh` để tạo file zip. Hoặc đẩy repo lên GitHub ở chế độ private.
2. Tải file zip lên `MyDrive/BrainDecoding/code/`.
3. Mở `colab/P1_EEG_00_data.ipynb` trên Colab, chạy hết với `SUBJECTS = ['sub-01']`.
4. Gửi lại cho em: dòng trong `results/pilot/profile.csv`, sai số đổi kiểu trong meta, và hai file CSV về category.
5. Chạy `colab/P1_EEG_01_pipeline_check.ipynb` để kiểm tra đường đi của shuffle, metric và PRR trên dữ liệu thật. Kết quả của notebook này chỉ để kiểm tra code.

## 6. Code đã có

| Module | Nội dung |
|---|---|
| `brain/data/things_eeg2.py` | Đọc file pickle gốc, chuyển sang float16 theo từng khúc, mở bằng memmap, cắt cửa sổ thời gian, trung bình n lần lặp |
| `brain/data/splits.py` | Tách validation theo concept, lưu và đọc lại từ JSON |
| `brain/data/categories.py` | Ghép concept với 27 category của THINGS, đếm số ảnh dùng được cho shuf-w |
| `brain/controls.py` | shuf-w, shuf-x, null-0, null-g (thống kê lấy từ train), pca-k (học trên train), nn-train |
| `common/metrics.py` | Two-way identification (Pearson hoặc cosine, có bản trong cùng category), top-k, PRR và PRR kèm khoảng tin cậy |
| `common/stats.py` | Bootstrap theo ảnh, tỉ số của hai trung bình, Wilcoxon có cặp, Holm |
| `brain/tier1.py` | Bảng điểm theo ảnh theo cột của PLAN.md và bảng PRR |
| `bench/profile.py` | Đo thời gian, RAM đỉnh, VRAM đỉnh, dung lượng thư mục |

Tất cả có test chạy trên CPU (`pytest -q`).

## 7. Điểm chị cần quyết định

1. **Bản tiền xử lý:** 63 kênh, 250 Hz (mặc định) hay 17 kênh, 100 Hz.
2. **Lưu trữ:** gói Drive 100 GB, hay giữ ít subject một lúc.
3. **Concept thuộc nhiều category:** trên toàn bộ 1.854 concept THINGS, bảng `category27_manual` cho 559 concept không thuộc category nào và 302 concept thuộc từ hai category trở lên. Nhiều trường hợp là category lồng nhau, ví dụ "apple" thuộc cả food và fruit. Mặc định đang loại các concept này khỏi shuf-w. Notebook 00 in bảng đếm cho 200 ảnh test với ba bảng category (manual, top-down, bottom-up) và hai cách xử lý, để chị chọn.
4. **Validation:** 74 concept (740 ảnh) tách theo concept, seed 0.

## 8. Việc chưa làm

- Wrapper train NICE và ATM với checkpoint chọn theo validation. Việc này cần clone hai repo gốc ở một commit cố định và ghi vào `third_party/VERSIONS.md`.
- Null và PCA ở mức tín hiệu: code control đã có, nhưng cần model đã train để chạy.
- Metric ảnh (PixCorr, SSIM, AlexNet, Inception, CLIP, EffNet, SwAV) và bước sinh ảnh.
- Nhánh fMRI.
