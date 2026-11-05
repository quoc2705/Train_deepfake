"""Module suy luận (Inference) cho mô hình Deepfake FGI (bản crop vùng miệng v5_mouth).

Hỗ trợ nạp checkpoint `.pth` / `.pth.tar`, xử lý video theo đúng quy trình của
`pbl6_1_preprocess_mouth.ipynb` và `pbl6_2_train_A_13k_mouth_v5.ipynb`.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import config as C
from model import My_Network
import preprocess as P


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Tự động chọn GPU CUDA nếu có, ngược lại dùng CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_checkpoint(
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[My_Network, torch.device, Dict[str, Any]]:
    """Nạp trọng số checkpoint vào mô hình My_Network.

    - Tự động tìm checkpoint trong `models/` nếu `ckpt_path` là None.
    - Hỗ trợ cả định dạng checkpoint theo epoch (`model_best_epoch*.pth.tar`)
      và file model cuối (`fgi_deepfake_detector_final.pth`).
    - Tự động xóa tiền tố `module.` nếu lưu từ `nn.DataParallel`.
    """
    if device is None:
        device = get_device()

    target_path = C.find_checkpoint_path(ckpt_path)
    if not os.path.isfile(target_path):
        raise FileNotFoundError(
            f"Không tìm thấy file checkpoint tại: {target_path}.\n"
            f"Vui lòng chọn file model (.pth hoặc .pth.tar) hoặc đặt vào thư mục: {C.MODELS_DIR}"
        )

    try:
        raw_checkpoint = torch.load(target_path, map_location=device, weights_only=False)
    except TypeError:
        raw_checkpoint = torch.load(target_path, map_location=device)

    info: Dict[str, Any] = {'path': target_path}
    cfg_dict: Dict[str, Any] = {}

    if isinstance(raw_checkpoint, dict) and 'state_dict' in raw_checkpoint:
        state_dict = raw_checkpoint['state_dict']
        for k in (
            'epoch', 'best_epoch', 'least_loss', 'best_valid_auc', 'max_valid_auc',
            'best_model_auc', 'best_valid_logloss', 'best_score', 'run_tag', 'auc', 'net'
        ):
            if k in raw_checkpoint:
                info[k] = raw_checkpoint[k]
        if isinstance(raw_checkpoint.get('config'), dict):
            cfg_dict = raw_checkpoint['config']
    elif isinstance(raw_checkpoint, dict):
        state_dict = raw_checkpoint
    else:
        raise ValueError(f"Dữ liệu checkpoint không hợp lệ trong {target_path}")

    cleaned_state_dict = {}
    for key, val in state_dict.items():
        clean_key = key[7:] if key.startswith('module.') else key
        cleaned_state_dict[clean_key] = val

    model = My_Network(
        network=cfg_dict.get('net', info.get('net', C.MODEL_NET)),
        with_attention=cfg_dict.get('with_attention', C.MODEL_WITH_ATTENTION),
        residual_conn=cfg_dict.get('residual_conn', C.MODEL_RESIDUAL_CONN),
        spatial_size=cfg_dict.get('spatial_size', C.MODEL_SPATIAL_SIZE),
    )

    load_result = model.load_state_dict(cleaned_state_dict, strict=False)
    info['missing_keys'] = load_result.missing_keys
    info['unexpected_keys'] = load_result.unexpected_keys
    info['all_matched'] = (len(load_result.missing_keys) == 0 and len(load_result.unexpected_keys) == 0)

    model.to(device)
    model.eval()
    return model, device, info


def predict_chunk_with_dist(
    model: My_Network,
    vid_tensor: torch.Tensor,
    aud_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """Dự đoán xác suất FAKE và khoảng cách Audio-Visual cho 1 chunk (30 frame + 48000 mẫu audio)."""
    if device is not None:
        vid_tensor = vid_tensor.to(device)
        aud_tensor = aud_tensor.to(device)

    with torch.no_grad():
        final_out, vid_aud_dist, _ = model(vid_tensor, aud_tensor)
        score = float(final_out[0].item())
        dist = float(torch.mean(vid_aud_dist[0]).item())
    return score, dist


def predict_chunk(
    model: My_Network,
    vid_tensor: torch.Tensor,
    aud_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> float:
    """Trả về xác suất FAKE (0.0 -> REAL, 1.0 -> FAKE) cho 1 đoạn 1 giây."""
    score, _ = predict_chunk_with_dist(model, vid_tensor, aud_tensor, device=device)
    return score


def predict_video(
    video_path: str,
    model: Optional[My_Network] = None,
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    max_chunks: int = C.DEFAULT_MAX_CHUNKS,
    max_seconds: float = C.DEFAULT_MAX_SECONDS,
    threshold: float = C.DEFAULT_THRESHOLD,
    crop_mode: str = C.DEFAULT_CROP_MODE,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Phân tích video theo đúng pipeline `predict_video` trong `pbl6_2_train_A_13k_mouth_v5.ipynb`."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")

    if device is None:
        device = get_device()

    ckpt_info: Dict[str, Any] = {}
    if model is None:
        model, device, ckpt_info = load_checkpoint(ckpt_path, device=device)

    yunet_ok, yunet_msg = P.ensure_yunet()
    lm_ok, lm_msg = (P.ensure_landmarker() if crop_mode == 'mouth_lm' else (False, 'Không dùng'))

    # 1. Đọc thông tin video
    vinfo = P.probe_video(video_path)
    if not vinfo.has_video:
        raise ValueError(f"File {video_path} không chứa luồng hình ảnh video.")
    if not vinfo.has_audio:
        raise ValueError("Video không có âm thanh -> không dự đoán được (mô hình FGI cần cả hình và tiếng).")

    # 2. Tách audio WAV 48kHz mono int16
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name

    try:
        full_audio, sample_rate = P.extract_audio(video_path, wav_path, seconds=max_seconds)
    finally:
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except OSError:
                pass

    # 3. Duyệt từng giây (30 frame/giây)
    chunk_results: List[Dict[str, Any]] = []
    skipped_chunks: List[Dict[str, Any]] = []
    detector = P.get_detector()
    total_lm_ok = 0
    total_lm_fb = 0

    total_est = int(min(vinfo.duration, max_seconds)) if vinfo.duration > 0 else int(max_seconds)
    total_est = max(1, total_est)

    for sec, frames in P.iter_video_chunks(video_path, vinfo, seconds=max_seconds, stop_flag=stop_flag):
        if stop_flag and stop_flag():
            break
        if progress_callback:
            progress_callback(sec + 1, total_est, f"Đang phân tích giây thứ {sec + 1}/{total_est}...")

        analysis = P.analyze_chunk(frames, detector=detector, crop_mode=crop_mode)
        total_lm_ok += analysis.lm_ok_count
        total_lm_fb += analysis.lm_fallback_count

        if analysis.reason != 'ok' or analysis.box is None:
            skipped_chunks.append({
                'sec': sec,
                'reason': analysis.reason,
                'reason_text': C.REASON_TEXT.get(analysis.reason, analysis.reason),
                'cut_score': analysis.cut_score,
            })
            continue

        crops = P.crop_chunk(frames, analysis.box)
        vid_tensor = P.frames_to_tensor(crops)

        sec_audio = P.slice_audio(full_audio, sec, sample_rate=sample_rate)
        aud_tensor = P.audio_to_tensor(sec_audio)

        score, dist = predict_chunk_with_dist(model, vid_tensor, aud_tensor, device=device)
        label = P.label_for(score, threshold)

        # Lưu 4 ảnh crop mẫu (frame 0, 10, 20, 29 giống preview_crops ở notebook 1) và 1 ảnh overlay
        preview_ids = (0, 10, 20, 29)
        preview_crops = [crops[k] for k in preview_ids if k < len(crops)]
        overlay_frame = P.draw_overlay(frames[15], analysis.box, score, threshold, 'ok', analysis.faces)

        chunk_results.append({
            'sec': sec,
            'score': score,
            'distance': dist,
            'label': label,
            'box': analysis.box,
            'cut_score': analysis.cut_score,
            'lm_ok_count': analysis.lm_ok_count,
            'lm_fallback_count': analysis.lm_fallback_count,
            'preview_crops': preview_crops,
            'overlay_frame': overlay_frame,
        })

        if max_chunks and len(chunk_results) >= max_chunks:
            break

    scores = [c['score'] for c in chunk_results]
    dists = [c['distance'] for c in chunk_results]
    overall_score = float(np.mean(scores)) if scores else None
    overall_dist = float(np.mean(dists)) if dists else None
    overall_label = P.label_for(overall_score, threshold) if overall_score is not None else 'N/A'

    reason_counts: Dict[str, int] = {}
    for s in skipped_chunks:
        r_text = s['reason_text']
        reason_counts[r_text] = reason_counts.get(r_text, 0) + 1

    return {
        'video_path': video_path,
        'video_info': {
            'width': vinfo.width,
            'height': vinfo.height,
            'fps': vinfo.fps,
            'duration': vinfo.duration,
        },
        'checkpoint_info': ckpt_info,
        'crop_mode': crop_mode,
        'detector_status': yunet_msg,
        'landmarker_status': lm_msg,
        'lm_stats': {'lm_ok': total_lm_ok, 'lm_fallback': total_lm_fb},
        'chunks': chunk_results,
        'valid_chunks_count': len(chunk_results),
        'skipped_chunks': skipped_chunks,
        'skipped_count': len(skipped_chunks),
        'reason_counts': reason_counts,
        'overall_score': overall_score,
        'overall_distance': overall_dist,
        'overall_label': overall_label,
        'threshold': threshold,
        'device': str(device),
    }

