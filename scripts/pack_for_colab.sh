#!/usr/bin/env bash
# Đóng gói code đã commit thành file zip để tải lên MyDrive/BrainDecoding/code/.
# Chỉ lấy file được git theo dõi, nên không bao giờ chứa data/, checkpoint hay ảnh.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! git diff --quiet HEAD; then
  echo "Cảnh báo: có thay đổi chưa commit; file zip chỉ chứa bản đã commit." >&2
fi
out="brain_decoding_$(date +%Y%m%d_%H%M)_$(git rev-parse --short HEAD).zip"
git archive --format=zip -o "$out" HEAD
echo "Đã tạo $out. Tải file này lên Google Drive: MyDrive/BrainDecoding/code/"
