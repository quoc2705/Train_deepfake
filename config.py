"""Hằng số tiền xử lý — ĐỒNG BỘ 100% với `pbl6_1_preprocess_mouth.ipynb` và `pbl6_2_train_A_13k_mouth_v5.ipynb`.

Không tự ý thay đổi các hằng số ở mục "Tiền xử lý" để đảm bảo dữ liệu suy luận
khớp hoàn toàn với dữ liệu lúc huấn luyện.
"""
from __future__ import annotations

import os

# ============================ Tiền xử lý (GIỐNG NOTEBOOK) ============================
FPS = 30
CHUNK_FRAMES = 30
AUDIO_SR = 48000
AUDIO_SAMPLES = 48000          # mono, int16 (1 giây = 48000 mẫu)
SCAN_SECONDS = 8               # mặc định trong notebook
FRAME_SHORT_SIDE = 540         # cạnh ngắn của frame trung gian dùng để dò mặt / crop miệng
CUT_THRESH = 0.45              # khoảng cách Bhattacharyya > 0.45 => có chuyển cảnh => loại đoạn
FACE_SAMPLE_IDX = (0, 7, 15, 22, 29)
MIN_FACE_FRAC = 0.6            # cần thấy mặt ở ít nhất 60% số frame mẫu (3/5 frame)

# Chế độ crop:
# - 'mouth_lm': YuNet tìm mặt + MediaPipe lấy 40 điểm landmark môi (tự fallback về 'mouth' nếu frame thiếu landmark)
# - 'mouth'   : Ô vuông quanh miệng theo kích thước mặt + 2 mép miệng của YuNet (không cần MediaPipe)
# - 'face'    : Crop cả khuôn mặt
DEFAULT_CROP_MODE = 'mouth_lm'

MIN_FACE_PX_MOUTH = 64         # cạnh mặt tối thiểu trên frame trung gian khi crop miệng
MIN_FACE_PX_FACE = 48          # cạnh mặt tối thiểu khi crop cả mặt
MAX_UNION_RATIO = 2.0          # vùng chứa tất cả box mẫu / kích thước trung bình > 2.0 => mặt di chuyển nhiều
FACE_MARGIN = 1.4              # hệ số mở rộng khi CROP_MODE = 'face'
MOUTH_SCALE = 0.7              # ('mouth') cạnh ô vuông quanh miệng = 0.7 * cạnh mặt
MOUTH_Y_SHIFT = 0.05           # đẩy tâm ô crop xuống dưới 5% cạnh ô để lấy thêm hàm/cằm
MOUTH_MARGIN = 1.1             # hệ số mở rộng sau khi hợp nhất các ô miệng trong 1 đoạn
LIP_SCALE = 1.6                # ('mouth_lm') cạnh ô = 1.6 * bề rộng môi đo từ 40 điểm landmark
LIP_MIN_FACE_FRAC = 0.5        # ('mouth_lm') ô không nhỏ hơn 50% cạnh mặt (tránh quá chặt khi chúm môi)
OUT_SIZE = 224
JPEG_QUALITY = 95

# Chuẩn hóa ảnh (test_transform của notebook 2)
IMG_MEAN = (0.5, 0.5, 0.5)
IMG_STD = (0.5, 0.5, 0.5)

# Tham số YuNet (giống notebook 1 & 2)
YUNET_URL = (
    'https://github.com/opencv/opencv_zoo/raw/main/models/'
    'face_detection_yunet/face_detection_yunet_2023mar.onnx'
)
YUNET_FILENAME = 'face_detection_yunet_2023mar.onnx'
YUNET_INPUT_SIZE = (320, 320)
YUNET_SCORE_THRESH = 0.7
YUNET_NMS_THRESH = 0.3
YUNET_TOP_K = 5000

# Tham số MediaPipe Face Landmarker
LANDMARKER_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'face_landmarker/face_landmarker/float16/1/face_landmarker.task'
)
LANDMARKER_FILENAME = 'face_landmarker.task'

# Kiến trúc model (Config của notebook 2)
MODEL_NET = 'resnet18'
MODEL_WITH_ATTENTION = True
MODEL_RESIDUAL_CONN = False
MODEL_SPATIAL_SIZE = 28

# ============================ Ứng dụng ============================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
YUNET_PATH = os.path.join(MODELS_DIR, YUNET_FILENAME)
LANDMARKER_PATH = os.path.join(MODELS_DIR, LANDMARKER_FILENAME)
DEFAULT_CHECKPOINT_FILENAME = 'model_best_epoch23.pth.tar'
DEFAULT_CHECKPOINT_PATH = os.path.join(MODELS_DIR, DEFAULT_CHECKPOINT_FILENAME)


def find_checkpoint_path(custom_path: str | None = None) -> str:
    """Tìm file checkpoint model: custom_path -> DEFAULT_CHECKPOINT_PATH -> file .pth/.tar bất kỳ trong models/."""
    if custom_path and os.path.isfile(custom_path):
        return custom_path
    if os.path.isfile(DEFAULT_CHECKPOINT_PATH):
        return DEFAULT_CHECKPOINT_PATH
    if os.path.isdir(MODELS_DIR):
        for fname in sorted(os.listdir(MODELS_DIR)):
            fpath = os.path.join(MODELS_DIR, fname)
            if os.path.isfile(fpath) and any(fname.endswith(ext) for ext in ('.pth', '.tar', '.pt')):
                return fpath
    return DEFAULT_CHECKPOINT_PATH


DEFAULT_THRESHOLD = 0.5        # điểm FAKE >= ngưỡng => FAKE
DEFAULT_MAX_SECONDS = 12       # giống predict_video trong notebook 2 (scan_seconds=12)
DEFAULT_MAX_CHUNKS = 10        # giống predict_video trong notebook 2 (max_chunks=10)

# Nhãn lý do loại đoạn -> tiếng Việt
REASON_TEXT = {
    'ok': 'Hợp lệ',
    'chuyen_canh': 'Chuyển cảnh (scene cut)',
    'khong_co_mat': 'Không đủ mặt hợp lệ (< 60% frame mẫu)',
    'mat_di_chuyen_nhieu': 'Mặt/miệng di chuyển nhiều hoặc đổi người',
    'video_khong_audio': 'Video không có âm thanh',
    'thieu_audio': 'Thiếu âm thanh',
    'thieu_frame': 'Thiếu frame (< 30 frame)',
    'doc_frame_loi': 'Lỗi đọc frame',
}

COLOR_REAL = '#2ecc71'
COLOR_FAKE = '#e74c3c'
COLOR_NA = '#8a8f98'

