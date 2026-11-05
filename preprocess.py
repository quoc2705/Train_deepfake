"""Tiền xử lý video & âm thanh — tái hiện ĐÚNG pipeline của `pbl6_1_preprocess_mouth.ipynb`
và `pbl6_2_train_A_13k_mouth_v5.ipynb`:
- Đổi về 30 fps, tách đoạn 1 giây (30 frame + 48000 mẫu audio WAV mono 48kHz int16).
- Phát hiện chuyển cảnh bằng khoảng cách Bhattacharyya trên histogram HSV (ngưỡng 0.45).
- Dò mặt ở 5 frame mẫu (0, 7, 15, 22, 29) bằng YuNet (kèm tâm miệng từ 2 mép miệng YuNet).
- Crop vùng MIỆNG (`mouth_lm`: MediaPipe 40 điểm môi + fallback YuNet `mouth`; hoặc `mouth`; hoặc `face`),
  hợp nhất thành 1 ô vuông cố định cho cả 30 frame trong đoạn rồi resize về 224x224.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

import config as C
from model import min_max_normalize

# (x, y, w, h, mouth_cx, mouth_cy)
FaceDet = Tuple[float, float, float, float, float, float]
# (x, y, w, h)
Box = Tuple[float, float, float, float]
# (cx, cy, side) trên frame trung gian
CropBox = Tuple[float, float, float]

_NO_WINDOW = 0x08000000 if sys.platform == 'win32' else 0


# ============================================================================
# Frame trung gian
# ============================================================================
def intermediate_size(W: int, H: int) -> Tuple[int, int]:
    """Kích thước frame trung gian, công thức y hệt `extract_frames_audio` của notebook."""
    if W > 0 and H > 0 and min(W, H) > C.FRAME_SHORT_SIDE:
        f = C.FRAME_SHORT_SIDE / min(W, H)
        W, H = int(round(W * f / 2) * 2), int(round(H * f / 2) * 2)
    return W, H


def half_size(img: np.ndarray) -> np.ndarray:
    """Tương đương `cv2.imread(..., IMREAD_REDUCED_COLOR_2)`: giải mã kích thước 1/2."""
    h, w = img.shape[:2]
    return cv2.resize(img, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)


# ============================================================================
# Phát hiện chuyển cảnh (Scene-cut filter)
# ============================================================================
def frame_hist(small: np.ndarray) -> np.ndarray:
    t = cv2.resize(small, (64, 36), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(h, h, 1, 0, cv2.NORM_L1)
    return h


def has_scene_cut(smalls: Sequence[np.ndarray]) -> Tuple[bool, float]:
    """Trả về (có_chuyển_cảnh, khoảng_cách_Bhattacharyya_lớn_nhất)."""
    prev, worst = None, 0.0
    for s in smalls:
        h = frame_hist(s)
        if prev is not None:
            worst = max(worst, cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA))
        prev = h
    return worst > C.CUT_THRESH, float(worst)


# ============================================================================
# Bộ dò mặt: YuNet (kèm tọa độ 2 mép miệng) -> fallback Haar cascade
# ============================================================================
_yunet_lock = threading.Lock()
_yunet_status: Optional[Tuple[bool, str]] = None


def ensure_yunet(download: bool = True) -> Tuple[bool, str]:
    """Đảm bảo có file YuNet trong `models/`. Trả về (ok, thông_báo)."""
    global _yunet_status
    with _yunet_lock:
        if _yunet_status is not None:
            return _yunet_status
        if not hasattr(cv2, 'FaceDetectorYN'):
            _yunet_status = (False, 'OpenCV không hỗ trợ FaceDetectorYN -> dùng Haar cascade')
            return _yunet_status
        path = C.YUNET_PATH
        try:
            if not (os.path.exists(path) and os.path.getsize(path) > 100_000):
                if not download:
                    raise FileNotFoundError(path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + '.part'
                urllib.request.urlretrieve(C.YUNET_URL, tmp)
                os.replace(tmp, path)
            if os.path.getsize(path) <= 100_000:
                raise IOError('File YuNet quá nhỏ hoặc bị hỏng')
            cv2.FaceDetectorYN.create(
                path, '', C.YUNET_INPUT_SIZE, C.YUNET_SCORE_THRESH, C.YUNET_NMS_THRESH, C.YUNET_TOP_K
            )
            _yunet_status = (True, 'YuNet')
        except Exception as e:  # noqa: BLE001
            _yunet_status = (False, f'Haar cascade (không nạp được YuNet: {e})')
        return _yunet_status


class FaceDetector:
    """YuNet trả về (x, y, w, h, tam_mieng_x, tam_mieng_y) giống hệt `_FaceDetector` trong notebook."""

    def __init__(self) -> None:
        self.yunet = None
        self.haar = None
        ok, _ = ensure_yunet()
        if ok:
            self.yunet = cv2.FaceDetectorYN.create(
                C.YUNET_PATH, '', C.YUNET_INPUT_SIZE,
                C.YUNET_SCORE_THRESH, C.YUNET_NMS_THRESH, C.YUNET_TOP_K
            )
        else:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    @property
    def name(self) -> str:
        return 'YuNet' if self.yunet is not None else 'Haar cascade'

    def detect(self, bgr: np.ndarray) -> List[FaceDet]:
        """Trả về list (x, y, w, h, tam_mieng_x, tam_mieng_y) theo tọa độ của ảnh đầu vào."""
        h, w = bgr.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(bgr)
            if faces is None:
                return []
            # f = [x, y, w, h, mat_phai(2), mat_trai(2), mui(2), mep_mieng_phai(2), mep_mieng_trai(2), score]
            return [
                (
                    float(f[0]), float(f[1]), float(f[2]), float(f[3]),
                    (float(f[10]) + float(f[12])) / 2.0,
                    (float(f[11]) + float(f[13])) / 2.0,
                )
                for f in faces
            ]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
        out: List[FaceDet] = []
        for (x, y, fw, fh) in found:
            out.append((float(x), float(y), float(fw), float(fh), float(x) + fw / 2.0, float(y) + 0.78 * fh))
        return out


_tls = threading.local()


def get_detector() -> FaceDetector:
    if not hasattr(_tls, 'det'):
        _tls.det = FaceDetector()
    return _tls.det


def pick_face(faces: List[FaceDet], prev_center: Optional[Tuple[float, float]]) -> Optional[FaceDet]:
    if not faces:
        return None
    if prev_center is None:
        return max(faces, key=lambda f: f[2] * f[3])
    return min(faces, key=lambda f: (f[0] + f[2] / 2 - prev_center[0]) ** 2 + (f[1] + f[3] / 2 - prev_center[1]) ** 2)


# ============================================================================
# MediaPipe Face Landmarker (40 điểm môi) + Fallback YuNet Mouth Box
# ============================================================================
_lm_lock = threading.Lock()
_lm_status: Optional[Tuple[bool, str]] = None


def _lip_indices() -> List[int]:
    from mediapipe.tasks.python import vision
    conns = vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
    return sorted({i for c in conns for i in (c.start, c.end)})


def _get_landmarker():
    if not hasattr(_tls, 'lm'):
        from mediapipe.tasks.python import vision, BaseOptions
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=C.LANDMARKER_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        _tls.lm = vision.FaceLandmarker.create_from_options(opts)
        _tls.lip_idx = np.array(_lip_indices())
    return _tls.lm, _tls.lip_idx


def ensure_landmarker(download: bool = True) -> Tuple[bool, str]:
    """Kiểm tra MediaPipe Face Landmarker. Nếu chưa có mediapipe hoặc model thì báo lý do để fallback."""
    global _lm_status
    with _lm_lock:
        if _lm_status is not None:
            return _lm_status
        try:
            import mediapipe  # noqa: F401
            path = C.LANDMARKER_PATH
            if not (os.path.exists(path) and os.path.getsize(path) > 1_000_000):
                if not download:
                    raise FileNotFoundError(path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = path + '.part'
                urllib.request.urlretrieve(C.LANDMARKER_URL, tmp)
                os.replace(tmp, path)
            if os.path.getsize(path) <= 1_000_000:
                raise IOError('File face_landmarker.task không hợp lệ')
            _get_landmarker()
            _lm_status = (True, 'MediaPipe Face Landmarker (40 điểm môi)')
        except Exception as e:  # noqa: BLE001
            _lm_status = (False, f'YuNet Mouth Fallback (MediaPipe chưa khả dụng: {e})')
        return _lm_status


def _lip_pts(rgb: np.ndarray) -> Optional[np.ndarray]:
    """Trả về mảng (40, 2) tọa độ chuẩn hóa (0..1) của các điểm môi, hoặc None."""
    import mediapipe as mp
    lm, idx = _get_landmarker()
    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return None
    p = res.face_landmarks[0]
    return np.array([(p[i].x, p[i].y) for i in idx], dtype=np.float32)


def _lip_box_from_pts(pts: Optional[np.ndarray], crop_rect_xywh: Tuple[int, int, int, int], f: FaceDet) -> Optional[Box]:
    """Tính ô vuông (x, y, w, h) quanh môi từ 40 điểm landmark giống hệt `_lip_box_from_pts` notebook."""
    if pts is None:
        return None
    x1, y1, cw, ch = crop_rect_xywh
    px = pts[:, 0] * cw + x1
    py = pts[:, 1] * ch + y1
    lx1, lx2, ly1, ly2 = float(px.min()), float(px.max()), float(py.min()), float(py.max())
    lip_w = lx2 - lx1
    face_side = 0.5 * (f[2] + f[3])
    cx, cy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
    if not (0.2 * face_side <= lip_w <= 0.8 * face_side):
        return None
    if ((cx - f[4]) ** 2 + (cy - f[5]) ** 2) ** 0.5 > 0.25 * face_side:
        return None
    side = max(C.LIP_SCALE * lip_w, C.LIP_MIN_FACE_FRAC * face_side)
    cy = cy + C.MOUTH_Y_SHIFT * side
    return (cx - side / 2.0, cy - side / 2.0, side, side)


def _lip_box(full_bgr: np.ndarray, f: FaceDet) -> Optional[Box]:
    H, W = full_bgr.shape[:2]
    x, y, w, h = f[:4]
    m = 0.25 * max(w, h)
    x1, y1 = int(max(0, x - m)), int(max(0, y - m))
    x2, y2 = int(min(W, x + w + m)), int(min(H, y + h + m))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    pts = _lip_pts(cv2.cvtColor(full_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
    return _lip_box_from_pts(pts, (x1, y1, x2 - x1, y2 - y1), f)


def _mouth_box(f: FaceDet) -> Box:
    """Từ (x, y, w, h, mouth_cx, mouth_cy) -> ô vuông (x, y, side, side) quanh miệng (YuNet)."""
    x, y, w, h, mx, my = f
    side = C.MOUTH_SCALE * 0.5 * (w + h)
    cy = my + C.MOUTH_Y_SHIFT * side
    return (mx - side / 2.0, cy - side / 2.0, side, side)


def sample_box(frame_bgr: np.ndarray, f: FaceDet, crop_mode: str = C.DEFAULT_CROP_MODE) -> Tuple[Box, str]:
    """Ô crop cho 1 frame mẫu theo `crop_mode`. Trả về (box, nguồn_box)."""
    if crop_mode == 'mouth_lm':
        lm_ok, _ = ensure_landmarker()
        if lm_ok and frame_bgr is not None:
            b = _lip_box(frame_bgr, f)
            if b is not None:
                return b, 'lm_ok'
        return _mouth_box(f), 'lm_fallback'
    if crop_mode == 'mouth':
        return _mouth_box(f), 'mouth_yunet'
    return f[:4], 'face'


# ============================================================================
# Phân tích + crop 1 đoạn 30 frame
# ============================================================================
@dataclass
class ChunkAnalysis:
    box: Optional[CropBox]                 # (cx, cy, side) hoặc None nếu bị loại
    reason: str                            # 'ok' | 'chuyen_canh' | 'khong_co_mat' | 'mat_di_chuyen_nhieu' | ...
    cut_score: float = 0.0
    faces: List[FaceDet] = field(default_factory=list)
    sample_boxes: List[Box] = field(default_factory=list)
    lm_ok_count: int = 0
    lm_fallback_count: int = 0


def analyze_chunk(
    frames: Sequence[np.ndarray],
    detector: Optional[FaceDetector] = None,
    crop_mode: str = C.DEFAULT_CROP_MODE,
) -> ChunkAnalysis:
    """Phân tích đoạn 30 frame trung gian (BGR). Logic y hệt `analyze_chunk` trong 2 file ipynb."""
    if len(frames) != C.CHUNK_FRAMES or any(f is None for f in frames):
        return ChunkAnalysis(None, 'thieu_frame')
    smalls = [half_size(f) for f in frames]
    cut, worst = has_scene_cut(smalls)
    if cut:
        return ChunkAnalysis(None, 'chuyen_canh', worst)

    det = detector or get_detector()
    min_face_px = C.MIN_FACE_PX_MOUTH if crop_mode in ('mouth', 'mouth_lm') else C.MIN_FACE_PX_FACE
    boxes: List[Box] = []
    picked_faces: List[FaceDet] = []
    prev: Optional[Tuple[float, float]] = None
    lm_ok_cnt = 0
    lm_fb_cnt = 0

    for k in C.FACE_SAMPLE_IDX:
        # Nhân 2 tất cả 6 giá trị (x, y, w, h, mx, my) về tọa độ frame trung gian
        faces = [tuple(v * 2.0 for v in f) for f in det.detect(smalls[k])]
        faces = [f for f in faces if min(f[2], f[3]) >= min_face_px]
        f = pick_face(faces, prev)
        if f is not None:
            b, src = sample_box(frames[k], f, crop_mode=crop_mode)
            boxes.append(b)
            picked_faces.append(f)
            if src == 'lm_ok':
                lm_ok_cnt += 1
            elif src == 'lm_fallback':
                lm_fb_cnt += 1
            # Vẫn theo dõi theo tâm MẶT để không nhảy sang người khác
            prev = (f[0] + f[2] / 2.0, f[1] + f[3] / 2.0)

    if len(boxes) < C.MIN_FACE_FRAC * len(C.FACE_SAMPLE_IDX):
        return ChunkAnalysis(None, 'khong_co_mat', worst, picked_faces, boxes, lm_ok_cnt, lm_fb_cnt)

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    mean_side = float(np.mean([max(b[2], b[3]) for b in boxes]))
    extent = max(x2 - x1, y2 - y1)
    if extent > C.MAX_UNION_RATIO * mean_side:
        return ChunkAnalysis(None, 'mat_di_chuyen_nhieu', worst, picked_faces, boxes, lm_ok_cnt, lm_fb_cnt)

    margin = C.MOUTH_MARGIN if crop_mode in ('mouth', 'mouth_lm') else C.FACE_MARGIN
    crop_box: CropBox = ((x1 + x2) / 2.0, (y1 + y2) / 2.0, extent * margin)
    return ChunkAnalysis(crop_box, 'ok', worst, picked_faces, boxes, lm_ok_cnt, lm_fb_cnt)


def crop_rect(box: CropBox, H: int, W: int) -> Tuple[int, int, int]:
    """Ô vuông crop (x1, y1, s) đã clamp trong ảnh — giống `crop_chunk` của notebook."""
    cx, cy, side = box
    s = int(round(min(side, H, W)))
    x1 = int(round(cx - s / 2.0))
    y1 = int(round(cy - s / 2.0))
    x1 = max(0, min(x1, W - s))
    y1 = max(0, min(y1, H - s))
    return x1, y1, s


def crop_chunk(frames: Sequence[np.ndarray], box: CropBox, out_size: int = C.OUT_SIZE) -> List[np.ndarray]:
    """Crop cùng 1 ô vuông cho cả 30 frame, resize 224x224 (INTER_AREA). Trả về list BGR."""
    H, W = frames[0].shape[:2]
    x1, y1, s = crop_rect(box, H, W)
    return [
        cv2.resize(img[y1:y1 + s, x1:x1 + s], (out_size, out_size), interpolation=cv2.INTER_AREA)
        for img in frames
    ]


# ============================================================================
# Chuẩn hóa Tensor (test_transform của notebook 2)
# ============================================================================
_to_tensor = transforms.ToTensor()
_normalize = transforms.Normalize(mean=list(C.IMG_MEAN), std=list(C.IMG_STD))


def frames_to_tensor(crops: Sequence[np.ndarray]) -> torch.Tensor:
    """30 ảnh crop BGR 224x224 -> tensor (1, 1, 3, 30, 224, 224) như `predict_video` notebook 2."""
    t_seq = [_normalize(_to_tensor(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops]
    (Cc, H, W) = t_seq[0].size()
    t = torch.stack(t_seq, 0)
    return t.view(1, 1, len(crops), Cc, H, W).transpose(2, 3)


def fit_audio(chunk_audio: np.ndarray, n: int = C.AUDIO_SAMPLES) -> np.ndarray:
    """Pad 0 hoặc cắt về đúng 48000 mẫu, giữ nguyên dtype int16."""
    chunk_audio = np.asarray(chunk_audio)
    if chunk_audio.ndim > 1:
        chunk_audio = chunk_audio[:, 0]
    if len(chunk_audio) < n:
        chunk_audio = np.pad(chunk_audio, (0, n - len(chunk_audio)))
    elif len(chunk_audio) > n:
        chunk_audio = chunk_audio[:n]
    return chunk_audio


def audio_to_tensor(chunk_audio: np.ndarray) -> torch.Tensor:
    """48000 mẫu int16 -> tensor (1, 48000), chuẩn hóa min-max y hệt notebook."""
    chunk_audio = fit_audio(chunk_audio)
    if chunk_audio.dtype != np.int16:
        chunk_audio = chunk_audio.astype(np.int16)
    norm_audio = min_max_normalize(chunk_audio, int(chunk_audio.min()), int(chunk_audio.max()))
    norm_audio = torch.from_numpy(norm_audio.astype(float)).float()
    return ((norm_audio - 0.5) / 0.5).unsqueeze(0)


def slice_audio(full_audio: np.ndarray, sec: int, sample_rate: int = C.AUDIO_SR) -> np.ndarray:
    """Lấy audio của giây thứ `sec` và pad/cắt về 48000 mẫu."""
    return fit_audio(full_audio[int(sec * sample_rate):int((sec + 1) * sample_rate)])


# ============================================================================
# FFmpeg tách frame và audio
# ============================================================================
class FFmpegError(RuntimeError):
    pass


def get_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        raise FFmpegError(f'Không tìm thấy ffmpeg (cần cài gói imageio-ffmpeg): {e}') from e


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool
    has_video: bool
    inter_w: int
    inter_h: int


def probe_video(path: str) -> VideoInfo:
    """Đọc thông tin video bằng OpenCV và kiểm tra luồng audio bằng ffmpeg."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Không tìm thấy file video: {path}')
    cap = cv2.VideoCapture(path)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    r = subprocess.run(
        [get_ffmpeg_exe(), '-hide_banner', '-i', path],
        capture_output=True,
        creationflags=_NO_WINDOW,
    )
    err = r.stderr.decode('utf-8', 'replace')
    has_audio = bool(re.search(r'Stream #\S+.*?: Audio:', err))
    has_video = bool(re.search(r'Stream #\S+.*?: Video:', err))
    duration = 0.0
    m = re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)', err)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    elif fps > 0 and n > 0:
        duration = n / fps
    if (W <= 0 or H <= 0) and has_video:
        m = re.search(r'Video:.*?(\d{2,5})x(\d{2,5})', err)
        if m:
            W, H = int(m.group(1)), int(m.group(2))
    iw, ih = intermediate_size(W, H)
    return VideoInfo(W, H, fps, duration, has_audio, has_video, iw, ih)


def extract_audio(video_path: str, wav_path: str, seconds: float) -> Tuple[np.ndarray, int]:
    """Tách audio mono 48 kHz int16 bằng ffmpeg (đúng tham số notebook), trả về (audio, sr)."""
    from scipy.io import wavfile
    cmd = [
        get_ffmpeg_exe(), '-y', '-v', 'error', '-i', video_path, '-t', str(seconds),
        '-vn', '-ac', '1', '-acodec', 'pcm_s16le', '-ar', str(C.AUDIO_SR), wav_path
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=600, creationflags=_NO_WINDOW)
    if r.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) <= 44:
        msg = r.stderr.decode('utf-8', 'replace').strip()[-500:]
        if 'does not contain any stream' in msg or 'matches no streams' in msg:
            raise FFmpegError('Video không có âm thanh — mô hình FGI cần cả khẩu hình miệng và tiếng nói.')
        raise FFmpegError(f'ffmpeg lỗi khi tách audio: {msg or "không rõ nguyên nhân"}')
    sr, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


def iter_video_chunks(
    video_path: str,
    info: VideoInfo,
    seconds: float,
    stop_flag=None,
) -> Iterator[Tuple[int, List[np.ndarray]]]:
    """Đọc video ở 30 fps, frame trung gian, yield từng đoạn 1 giây (sec, 30 frame BGR)."""
    W, H = info.inter_w, info.inter_h
    if W <= 0 or H <= 0:
        raise FFmpegError('Không đọc được kích thước khung hình của video.')
    cmd = [
        get_ffmpeg_exe(), '-v', 'error', '-i', video_path, '-t', str(seconds),
        '-r', str(C.FPS), '-vf', f'scale={W}:{H}', '-threads', '1',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=W * H * 3 * 4, creationflags=_NO_WINDOW
    )
    fsize = W * H * 3
    sec = 0
    try:
        while True:
            if stop_flag is not None and stop_flag():
                break
            frames: List[np.ndarray] = []
            for _ in range(C.CHUNK_FRAMES):
                buf = proc.stdout.read(fsize)
                if len(buf) < fsize:
                    break
                frames.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy())
            if len(frames) < C.CHUNK_FRAMES:
                break
            yield sec, frames
            sec += 1
    finally:
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
        err = proc.stderr.read().decode('utf-8', 'replace') if proc.stderr else ''
        proc.wait()
        if sec == 0 and err.strip() and not (stop_flag and stop_flag()):
            raise FFmpegError(f'ffmpeg lỗi khi đọc video: {err.strip()[-500:]}')


def label_for(score: Optional[float], threshold: float) -> str:
    if score is None:
        return 'N/A'
    return 'FAKE' if score >= threshold else 'REAL'


def draw_overlay(
    frame: np.ndarray,
    box: Optional[CropBox],
    score: Optional[float],
    threshold: float,
    reason: str = 'ok',
    faces: Sequence[FaceDet] = (),
) -> np.ndarray:
    """Vẽ khung mặt (nét mảnh) + khung crop miệng (nét đậm) + điểm dự đoán lên frame trung gian."""
    img = frame.copy()
    H, W = img.shape[:2]
    for f in faces:
        x, y, w, h = f[:4]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (180, 180, 180), 1)
    if box is not None and score is not None:
        lab = label_for(score, threshold)
        color = (113, 204, 46) if lab == 'REAL' else (60, 76, 231)
        x1, y1, s = crop_rect(box, H, W)
        cv2.rectangle(img, (x1, y1), (x1 + s, y1 + s), color, 2)
        text = f'{lab}  P(fake)={score:.3f}'
    else:
        color = (152, 143, 138)
        text = 'Bo qua: ' + reason
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(img, (8, 8), (16 + tw, 18 + th), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 12 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return img

