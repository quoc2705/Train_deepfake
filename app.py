"""Điểm khởi chạy ứng dụng FGI Deepfake Tester — Bản Crop Vùng Miệng (v5_mouth).

Hỗ trợ cả giao diện Desktop GUI (PySide6), dòng lệnh (CLI) và kiểm thử tự động (Unit Test).

Cách sử dụng:
    # 1. Mở giao diện Desktop GUI (mặc định):
    .venv\\Scripts\\python.exe app.py

    # 2. Chạy kiểm thử tự động toàn bộ pipeline & checkpoint:
    .venv\\Scripts\\python.exe app.py --test

    # 3. Chạy kiểm tra video qua dòng lệnh (CLI):
    .venv\\Scripts\\python.exe app.py --cli --video path/to/video.mp4
    .venv\\Scripts\\python.exe app.py --cli --video path/to/video.mp4 --ckpt models/model_best_epoch23.pth.tar --threshold 0.5
"""
from __future__ import annotations

import argparse
import os
import sys
import unittest

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def run_tests() -> int:
    """Chạy bộ kiểm thử tự động pipeline."""
    print("=" * 65)
    print(" BẮT ĐẦU KIỂM THỬ PIPELINE CROP MIỆNG (tests/test_pipeline.py)")
    print("=" * 65)
    suite = unittest.defaultTestLoader.discover('tests', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def run_cli(args: argparse.Namespace) -> int:
    """Chạy phân tích video chế độ dòng lệnh (CLI)."""
    import config as C
    import inference as I

    video_path = args.video
    if not video_path:
        print("LỖI: Chế độ CLI yêu cầu tham số --video <đường_dẫn_video>")
        return 1

    if not os.path.isfile(video_path):
        print(f"LỖI: Không tìm thấy file video: {video_path}")
        return 1

    ckpt_path = args.ckpt or C.find_checkpoint_path()
    threshold = args.threshold

    print("=" * 70)
    print(" HỆ THỐNG KIỂM THỬ DEEPFAKE FGI — BẢN CROP VÙNG MIỆNG (v5_mouth)")
    print("=" * 70)
    print(f" Video đầu vào   : {video_path}")
    print(f" File Checkpoint : {ckpt_path}")
    print(f" Chế độ Crop     : {args.crop_mode}")
    print(f" Ngưỡng FAKE     : {threshold:.2f}")
    print(f" Đoạn tối đa     : {args.max_chunks} đoạn (quét tối đa {args.max_seconds:.0f}s đầu)")
    print("-" * 70)

    try:
        model, device, info = I.load_checkpoint(ckpt_path)
        print(f" Thiết bị chạy   : {device.type.upper()}")
        print(f" Khớp trọng số   : {'100% khớp' if info.get('all_matched') else 'Không khớp hoàn toàn'}")
        if 'epoch' in info:
            auc_val = info.get('best_model_auc', info.get('best_valid_auc', 'N/A'))
            print(f" Checkpoint Info : Epoch {info.get('epoch')}, Valid AUC: {auc_val}, Tag: {info.get('run_tag', 'N/A')}")
        print("-" * 70)
        print("Đang phân tích video...")

        result = I.predict_video(
            video_path=video_path,
            model=model,
            device=device,
            max_chunks=args.max_chunks,
            max_seconds=args.max_seconds,
            threshold=threshold,
            crop_mode=args.crop_mode,
        )

        vinfo = result['video_info']
        print(
            f"Thông tin video  : {vinfo['width']}x{vinfo['height']} @ {vinfo['fps']:.1f} fps, "
            f"thời lượng: {vinfo['duration']:.2f}s"
        )
        print(f"Bộ dò mặt        : {result.get('detector_status')}")
        print(f"Landmark môi     : {result.get('landmarker_status')} | Thống kê: {result.get('lm_stats')}")
        print("\n--- CHI TIẾT TỪNG ĐOẠN 1 GIÂY HỢP LỆ ---")

        chunks = result['chunks']
        if chunks:
            for c in chunks:
                sec = c['sec']
                score = c['score']
                dist = c['distance']
                lbl = c['label']
                print(f"  [Giây {sec:02d} - {sec + 1:02d}s] P(FAKE): {score:.4f} | A-V Dist: {dist:.4f} => {lbl}")
        else:
            print("  Không có đoạn nào đủ điều kiện hợp lệ.")

        skipped = result['skipped_chunks']
        if skipped:
            print("\n--- CÁC ĐOẠN BỊ LOẠI ---")
            for s in skipped:
                sec = s['sec']
                print(f"  [Giây {sec:02d} - {sec + 1:02d}s] Lý do: {s['reason_text']} (cut={s.get('cut_score', 0):.2f})")

        print("\n" + "=" * 70)
        print(" KẾT QUẢ TỔNG HỢP TOÀN VIDEO")
        print("=" * 70)
        if result['overall_score'] is not None:
            ov_score = result['overall_score']
            ov_label = result['overall_label']
            print(f" Điểm FAKE trung bình : {ov_score:.4f} ({ov_score * 100:.2f}%)")
            print(f" Kết luận cuối cùng   : {ov_label}")
        else:
            print(" Kết luận cuối cùng   : N/A (Video không có đoạn hợp lệ)")
        print(f" Đoạn hợp lệ          : {result['valid_chunks_count']} | Bị loại: {result['skipped_count']}")
        print("=" * 70)
        return 0

    except Exception as e:
        print(f"\n[LỖI] Quá trình phân tích thất bại: {e}")
        return 1


def run_gui() -> int:
    """Khởi chạy ứng dụng giao diện Desktop PySide6."""
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


def main():
    import config as C

    parser = argparse.ArgumentParser(description="FGI Deepfake Detector (Mouth Crop v5_mouth) CLI & GUI")
    parser.add_argument("--cli", action="store_true", help="Chạy chế độ dòng lệnh (CLI)")
    parser.add_argument("--video", type=str, default=None, help="Đường dẫn file video đầu vào")
    parser.add_argument("--ckpt", type=str, default=None, help="Đường dẫn file model checkpoint (.pth / .pth.tar)")
    parser.add_argument("--threshold", type=float, default=C.DEFAULT_THRESHOLD, help="Ngưỡng phát hiện FAKE (mặc định 0.5)")
    parser.add_argument("--crop-mode", type=str, default=C.DEFAULT_CROP_MODE, choices=["mouth_lm", "mouth", "face"],
                        help="Chế độ crop (mặc định: mouth_lm)")
    parser.add_argument("--max-chunks", type=int, default=C.DEFAULT_MAX_CHUNKS, help="Số đoạn 1s tối đa (mặc định 10)")
    parser.add_argument("--max-seconds", type=float, default=float(C.DEFAULT_MAX_SECONDS),
                        help="Thời gian quét tối đa (giây, mặc định 12)")
    parser.add_argument("--test", action="store_true", help="Chạy kiểm thử tự động pipeline")
    parser.add_argument("--gui", action="store_true", help="Mở giao diện đồ họa Desktop")

    args = parser.parse_args()

    if args.test:
        sys.exit(run_tests())
    elif args.cli or args.video:
        sys.exit(run_cli(args))
    else:
        try:
            sys.exit(run_gui())
        except Exception as e:
            print(f"Không thể mở giao diện đồ họa ({e}). Chuyển sang hướng dẫn dòng lệnh.")
            parser.print_help()
            sys.exit(1)


if __name__ == '__main__':
    main()

