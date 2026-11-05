"""Bộ kiểm thử tự động cho pipeline FGI Deepfake (Bản Crop Miệng v5_mouth):
1. Shape tensor video: (1, 1, 3, 30, 224, 224)
2. Shape tensor audio: (1, 48000) và xử lý an toàn audio im lặng
3. Hàm tính ô vuông crop miệng từ YuNet / MediaPipe (`_mouth_box`, `_lip_box_from_pts`)
4. Nạp checkpoint `models/model_best_epoch23.pth.tar` khớp 100% keys
5. Chạy forward pass qua model trả về xác suất FAKE hợp lệ trong [0, 1] và khoảng cách A-V
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

import config as C
import inference as I
import preprocess as P


class TestMouthDeepfakePipeline(unittest.TestCase):

    def test_01_video_tensor_shape(self):
        """Kiểm tra shape tensor video sau chuẩn hóa: (1, 1, 3, 30, 224, 224)."""
        dummy_crops = [np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(30)]
        tensor = P.frames_to_tensor(dummy_crops)
        expected_shape = (1, 1, 3, 30, 224, 224)
        self.assertEqual(tensor.shape, expected_shape)
        self.assertFalse(torch.isnan(tensor).any())

    def test_02_audio_tensor_shape_and_silence(self):
        """Kiểm tra shape audio (1, 48000), không NaN và audio im lặng không crash."""
        dummy_audio = np.random.randint(-3000, 3000, size=48000, dtype=np.int16)
        tensor = P.audio_to_tensor(dummy_audio)
        expected_shape = (1, 48000)
        self.assertEqual(tensor.shape, expected_shape)
        self.assertFalse(torch.isnan(tensor).any())

        silent_audio = np.zeros(48000, dtype=np.int16)
        silent_tensor = P.audio_to_tensor(silent_audio)
        self.assertEqual(silent_tensor.shape, expected_shape)
        self.assertFalse(torch.isnan(silent_tensor).any())

    def test_03_mouth_crop_box_math(self):
        """Kiểm tra công thức tính ô vuông crop miệng từ mặt và điểm môi."""
        # Giả lập khuôn mặt ở (100, 100, 200, 200) với tâm miệng tại (200, 250)
        face_det = (100.0, 100.0, 200.0, 200.0, 200.0, 250.0)
        mb = P._mouth_box(face_det)
        self.assertEqual(len(mb), 4)
        self.assertAlmostEqual(mb[2], C.MOUTH_SCALE * 200.0)
        self.assertAlmostEqual(mb[3], C.MOUTH_SCALE * 200.0)

        # Giả lập 40 điểm môi chuẩn hóa trong vùng crop (100, 100, 200, 200)
        pts = np.linspace(0.35, 0.65, 80, dtype=np.float32).reshape(40, 2)
        pts[:, 1] = np.linspace(0.70, 0.80, 40, dtype=np.float32)
        lb = P._lip_box_from_pts(pts, (100, 100, 200, 200), face_det)
        self.assertIsNotNone(lb)
        self.assertEqual(len(lb), 4)
        self.assertAlmostEqual(lb[2], lb[3])

    def test_04_load_checkpoint_100_percent_matched(self):
        """Kiểm tra checkpoint `model_best_epoch23.pth.tar` nạp khớp 100% key."""
        model, device, info = I.load_checkpoint()
        self.assertTrue(
            info['all_matched'],
            f"Checkpoint chưa khớp 100%! Missing: {info['missing_keys']}, Unexpected: {info['unexpected_keys']}"
        )
        self.assertEqual(len(info['missing_keys']), 0)
        self.assertEqual(len(info['unexpected_keys']), 0)

    def test_05_forward_pass(self):
        """Kiểm tra chạy forward pass qua model ra xác suất hợp lệ trong khoảng [0, 1]."""
        model, device, _ = I.load_checkpoint()
        dummy_video = torch.randn(1, 1, 3, 30, 224, 224, dtype=torch.float32, device=device)
        dummy_audio = torch.randn(1, 48000, dtype=torch.float32, device=device)

        score, dist = I.predict_chunk_with_dist(model, dummy_video, dummy_audio, device=device)
        self.assertIsInstance(score, float)
        self.assertIsInstance(dist, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertFalse(np.isnan(score))


if __name__ == '__main__':
    unittest.main()

