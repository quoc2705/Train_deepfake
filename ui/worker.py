"""Worker thread chạy phân tích video ở luồng nền để không làm đơ giao diện."""
from __future__ import annotations

import traceback
from typing import Optional

from PySide6.QtCore import QThread, Signal

import config as C
import inference as I
from model import My_Network


class VideoAnalysisWorker(QThread):
    progress = Signal(int, int, str)       # current_sec, total_sec, message
    finished = Signal(dict)                # result dict
    error = Signal(str)                    # error message

    def __init__(
        self,
        video_path: str,
        ckpt_path: str,
        model: Optional[My_Network] = None,
        device=None,
        threshold: float = C.DEFAULT_THRESHOLD,
        max_chunks: int = C.DEFAULT_MAX_CHUNKS,
        max_seconds: float = C.DEFAULT_MAX_SECONDS,
        crop_mode: str = C.DEFAULT_CROP_MODE,
    ):
        super().__init__()
        self.video_path = video_path
        self.ckpt_path = ckpt_path
        self.model = model
        self.device = device
        self.threshold = threshold
        self.max_chunks = max_chunks
        self.max_seconds = max_seconds
        self.crop_mode = crop_mode
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            def on_progress(cur: int, total: int, msg: str):
                if not self._is_cancelled:
                    self.progress.emit(cur, total, msg)

            result = I.predict_video(
                video_path=self.video_path,
                model=self.model,
                ckpt_path=self.ckpt_path,
                device=self.device,
                threshold=self.threshold,
                max_chunks=self.max_chunks,
                max_seconds=self.max_seconds,
                crop_mode=self.crop_mode,
                progress_callback=on_progress,
                stop_flag=lambda: self._is_cancelled,
            )
            if not self._is_cancelled:
                self.finished.emit(result)
        except Exception as e:
            err_msg = str(e)
            if not err_msg:
                err_msg = traceback.format_exc()
            self.error.emit(err_msg)

