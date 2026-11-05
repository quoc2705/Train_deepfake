"""Giao diện Desktop (PySide6) kiểm thử mô hình Deepfake FGI (Bản Crop Miệng v5_mouth)."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import config as C
import inference as I
import preprocess as P
from ui.worker import VideoAnalysisWorker

STYLE_SHEET = """
QWidget {
    background-color: #1a1b26;
    color: #e0e6ed;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2f334d;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: bold;
    color: #7aa2f7;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLineEdit, QComboBox, QSpinBox {
    background-color: #24283b;
    border: 1px solid #414868;
    border-radius: 6px;
    padding: 5px 10px;
    color: #ffffff;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
    border: 1px solid #7aa2f7;
}
QPushButton {
    background-color: #3d59a1;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #5472c4;
}
QPushButton:pressed {
    background-color: #2e437a;
}
QPushButton:disabled {
    background-color: #292e42;
    color: #565f89;
}
QProgressBar {
    border: 1px solid #414868;
    border-radius: 6px;
    text-align: center;
    background-color: #24283b;
    color: #ffffff;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #7aa2f7;
    border-radius: 5px;
}
QTableWidget {
    background-color: #24283b;
    border: 1px solid #414868;
    border-radius: 6px;
    gridline-color: #2f334d;
    selection-background-color: #364a82;
}
QHeaderView::section {
    background-color: #1f2335;
    color: #7aa2f7;
    padding: 6px;
    border: 1px solid #2f334d;
    font-weight: bold;
}
QSlider::groove:horizontal {
    border: 1px solid #414868;
    height: 6px;
    background: #24283b;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #7aa2f7;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 1px solid #7aa2f7;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
"""


def bgr_to_pixmap(bgr: np.ndarray, max_w: int, max_h: int) -> QPixmap:
    """Chuyển ảnh BGR (OpenCV) thành QPixmap đã scale vừa khung."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class MainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FGI Deepfake Detector — Kiểm thử Model Crop Vùng Miệng (v5_mouth)")
        self.resize(1100, 760)
        self.setStyleSheet(STYLE_SHEET)

        self.worker: Optional[VideoAnalysisWorker] = None
        self.loaded_model = None
        self.loaded_device = None
        self.loaded_ckpt_path: Optional[str] = None
        self.loaded_ckpt_info: Dict[str, Any] = {}
        self.last_result: Optional[Dict[str, Any]] = None

        self._init_ui()
        self._load_default_paths()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(14, 14, 14, 14)

        # Header
        title_label = QLabel("HỆ THỐNG KIỂM THỬ DEEPFAKE AUDIO-VISUAL FGI (CROP VÙNG MIỆNG)")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(15)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #7aa2f7; margin-bottom: 2px;")
        main_layout.addWidget(title_label)

        # 1. Cấu hình Model & Video
        file_group = QGroupBox("1. Import Model Checkpoint & Video Kiểm Tra")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(8)

        # Checkpoint Row
        ckpt_layout = QHBoxLayout()
        lbl_ckpt = QLabel("File Model (.pth/.tar):")
        lbl_ckpt.setFixedWidth(140)
        ckpt_layout.addWidget(lbl_ckpt)
        self.ckpt_edit = QLineEdit()
        self.ckpt_edit.setPlaceholderText("Chọn file model_best_epoch*.pth.tar hoặc fgi_deepfake_detector_final.pth...")
        ckpt_layout.addWidget(self.ckpt_edit, 1)

        self.btn_browse_ckpt = QPushButton("Chọn Model...")
        self.btn_browse_ckpt.clicked.connect(self._browse_ckpt)
        ckpt_layout.addWidget(self.btn_browse_ckpt)

        self.btn_load_ckpt = QPushButton("Nạp / Kiểm tra Model")
        self.btn_load_ckpt.setStyleSheet("background-color: #2ac3de; color: #1a1b26;")
        self.btn_load_ckpt.clicked.connect(self._manual_load_model)
        ckpt_layout.addWidget(self.btn_load_ckpt)
        file_layout.addLayout(ckpt_layout)

        # Model Info Label
        self.model_info_lbl = QLabel("Chưa nạp model.")
        self.model_info_lbl.setStyleSheet("color: #9ece6a; font-size: 12px; padding-left: 4px;")
        file_layout.addWidget(self.model_info_lbl)

        # Video Row
        vid_layout = QHBoxLayout()
        lbl_vid = QLabel("Video cần kiểm tra:")
        lbl_vid.setFixedWidth(140)
        vid_layout.addWidget(lbl_vid)
        self.video_edit = QLineEdit()
        self.video_edit.setPlaceholderText("Chọn file video có tiếng nói (.mp4, .avi, .mov, .mkv)...")
        vid_layout.addWidget(self.video_edit, 1)

        self.btn_browse_video = QPushButton("Chọn Video...")
        self.btn_browse_video.clicked.connect(self._browse_video)
        vid_layout.addWidget(self.btn_browse_video)
        file_layout.addLayout(vid_layout)

        # Params Row
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Chế độ Crop:"))
        self.crop_combo = QComboBox()
        self.crop_combo.addItem("mouth_lm — YuNet + MediaPipe Môi (Chuẩn v5_mouth)", "mouth_lm")
        self.crop_combo.addItem("mouth — YuNet 2 mép miệng (Không cần MediaPipe)", "mouth")
        self.crop_combo.addItem("face — Crop cả khuôn mặt (Bản cũ)", "face")
        params_layout.addWidget(self.crop_combo)

        params_layout.addSpacing(14)
        params_layout.addWidget(QLabel("Ngưỡng FAKE:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(5, 95)
        self.threshold_slider.setValue(int(C.DEFAULT_THRESHOLD * 100))
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        params_layout.addWidget(self.threshold_slider, 1)

        self.threshold_val_lbl = QLabel(f"{C.DEFAULT_THRESHOLD:.2f}")
        self.threshold_val_lbl.setStyleSheet("font-weight: bold; color: #7aa2f7; min-width: 36px;")
        params_layout.addWidget(self.threshold_val_lbl)

        params_layout.addSpacing(14)
        params_layout.addWidget(QLabel("Số đoạn tối đa (1s):"))
        self.chunks_spin = QSpinBox()
        self.chunks_spin.setRange(1, 60)
        self.chunks_spin.setValue(C.DEFAULT_MAX_CHUNKS)
        params_layout.addWidget(self.chunks_spin)

        params_layout.addSpacing(10)
        params_layout.addWidget(QLabel("Quét tối đa (giây):"))
        self.seconds_spin = QSpinBox()
        self.seconds_spin.setRange(1, 300)
        self.seconds_spin.setValue(int(C.DEFAULT_MAX_SECONDS))
        params_layout.addWidget(self.seconds_spin)

        file_layout.addLayout(params_layout)
        main_layout.addWidget(file_group)

        # 2. Nút điều khiển & Thanh tiến trình
        action_layout = QHBoxLayout()
        self.btn_start = QPushButton("▶ BẮT ĐẦU KIỂM TRA VIDEO")
        self.btn_start.setFixedHeight(38)
        self.btn_start.setStyleSheet("background-color: #7aa2f7; color: #1a1b26; font-size: 14px;")
        self.btn_start.clicked.connect(self._start_analysis)
        action_layout.addWidget(self.btn_start, 1)

        self.btn_stop = QPushButton("■ Dừng lại")
        self.btn_stop.setFixedHeight(38)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #f7768e; color: #1a1b26;")
        self.btn_stop.clicked.connect(self._stop_analysis)
        action_layout.addWidget(self.btn_stop)
        main_layout.addLayout(action_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Sẵn sàng.")
        self.status_lbl.setStyleSheet("color: #a9b1d6; font-style: italic;")
        main_layout.addWidget(self.status_lbl)

        # 3. Kết quả tổng hợp
        self.summary_group = QGroupBox("2. Kết Quả Đánh Giá Tổng Hợp Toàn Video")
        summary_layout = QHBoxLayout(self.summary_group)

        self.result_badge = QLabel("CHƯA KIỂM TRA")
        self.result_badge.setAlignment(Qt.AlignCenter)
        self.result_badge.setFixedSize(175, 72)
        self.result_badge.setStyleSheet(
            "background-color: #292e42; color: #a9b1d6; border-radius: 8px; font-size: 16px; font-weight: bold;"
        )
        summary_layout.addWidget(self.result_badge)

        self.summary_detail = QLabel("Hãy chọn Model và Video rồi nhấn 'Bắt đầu kiểm tra video'.")
        self.summary_detail.setWordWrap(True)
        self.summary_detail.setStyleSheet("font-size: 13px; padding-left: 8px;")
        summary_layout.addWidget(self.summary_detail, 1)
        main_layout.addWidget(self.summary_group)

        # 4. Bảng chi tiết từng giây + Khung xem trực quan vùng Crop Miệng
        bottom_splitter = QSplitter(Qt.Horizontal)

        # Trái: Bảng từng đoạn 1s
        table_group = QGroupBox("3. Chi Tiết Từng Đoạn 1 Giây (Click vào dòng để xem ảnh Crop Miệng)")
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "Đoạn (Giây)", "Xác suất FAKE", "Khoảng cách A-V", "Kết luận", "Trạng thái / Lý do"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_table_row_selected)
        table_layout.addWidget(self.table)
        bottom_splitter.addWidget(table_group)

        # Phải: Xem trước Khung hình & 4 frame Crop Miệng (0, 10, 20, 29)
        preview_group = QGroupBox("4. Xem Trực Quan Vùng Crop Miệng (224x224)")
        preview_layout = QVBoxLayout(preview_group)

        self.preview_title_lbl = QLabel("Chưa có đoạn hợp lệ nào được chọn.")
        self.preview_title_lbl.setAlignment(Qt.AlignCenter)
        self.preview_title_lbl.setStyleSheet("color: #7dcfff; font-weight: bold;")
        preview_layout.addWidget(self.preview_title_lbl)

        self.overlay_img_lbl = QLabel("Khung hình gốc + Ô vuông Crop")
        self.overlay_img_lbl.setAlignment(Qt.AlignCenter)
        self.overlay_img_lbl.setMinimumSize(280, 165)
        self.overlay_img_lbl.setStyleSheet("background-color: #16161e; border: 1px solid #2f334d; border-radius: 6px;")
        preview_layout.addWidget(self.overlay_img_lbl, 1)

        crops_row_lbl = QLabel("4 Frame mẫu đưa vào Model (Frame 0, 10, 20, 29):")
        crops_row_lbl.setStyleSheet("font-size: 12px; color: #a9b1d6;")
        preview_layout.addWidget(crops_row_lbl)

        crops_layout = QHBoxLayout()
        self.crop_labels: List[QLabel] = []
        for idx_frame in (0, 10, 20, 29):
            lbl = QLabel(f"F.{idx_frame}")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(96, 96)
            lbl.setStyleSheet("background-color: #16161e; border: 1px solid #414868; border-radius: 4px;")
            crops_layout.addWidget(lbl)
            self.crop_labels.append(lbl)
        preview_layout.addLayout(crops_layout)

        bottom_splitter.addWidget(preview_group)
        bottom_splitter.setStretchFactor(0, 3)
        bottom_splitter.setStretchFactor(1, 2)
        main_layout.addWidget(bottom_splitter, 1)

    def _load_default_paths(self):
        default_ckpt = C.find_checkpoint_path()
        if os.path.isfile(default_ckpt):
            self.ckpt_edit.setText(default_ckpt)
            self._manual_load_model(silent=True)

    def _on_threshold_changed(self, val: int):
        t = val / 100.0
        self.threshold_val_lbl.setText(f"{t:.2f}")
        # Nếu đã có kết quả phân tích, cập nhật lại nhãn theo ngưỡng mới ngay lập tức
        if self.last_result is not None:
            self.last_result['threshold'] = t
            for c in self.last_result['chunks']:
                c['label'] = P.label_for(c['score'], t)
            if self.last_result['overall_score'] is not None:
                self.last_result['overall_label'] = P.label_for(self.last_result['overall_score'], t)
            self._render_result(self.last_result)

    def _browse_ckpt(self):
        start_dir = C.MODELS_DIR if os.path.isdir(C.MODELS_DIR) else C.PROJECT_DIR
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Chọn file Model Checkpoint", start_dir,
            "PyTorch Checkpoint (*.pth *.pth.tar *.pt);;Tất cả file (*.*)"
        )
        if fpath:
            self.ckpt_edit.setText(fpath)
            self._manual_load_model(silent=False)

    def _browse_video(self):
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Chọn file Video kiểm tra", C.PROJECT_DIR,
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;Tất cả file (*.*)"
        )
        if fpath:
            self.video_edit.setText(fpath)

    def _manual_load_model(self, silent: bool = False):
        ckpt_path = self.ckpt_edit.text().strip()
        if not ckpt_path or not os.path.isfile(ckpt_path):
            if not silent:
                QMessageBox.warning(self, "Không tìm thấy Model", f"Đường dẫn không tồn tại:\n{ckpt_path}")
            return
        try:
            self.status_lbl.setText("Đang nạp checkpoint model...")
            model, device, info = I.load_checkpoint(ckpt_path)
            self.loaded_model = model
            self.loaded_device = device
            self.loaded_ckpt_path = ckpt_path
            self.loaded_ckpt_info = info

            ep = info.get('epoch', 'N/A')
            auc = info.get('best_model_auc', info.get('best_valid_auc', info.get('auc', None)))
            auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) else "N/A"
            score = info.get('best_score', None)
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            tag = info.get('run_tag', 'N/A')
            matched = "Khớp 100% trọng số" if info.get('all_matched') else f"Thiếu {len(info.get('missing_keys', []))} key"

            yunet_ok, yunet_msg = P.ensure_yunet(download=False)
            lm_ok, lm_msg = P.ensure_landmarker(download=False)

            self.model_info_lbl.setText(
                f"✔ Đã nạp: {os.path.basename(ckpt_path)} | Epoch: {ep} | Valid AUC: {auc_str} | "
                f"Select Score: {score_str} | Tag: {tag} | {matched} | Thiết bị: {device.type.upper()} | "
                f"Dò mặt: {yunet_msg}"
            )
            self.model_info_lbl.setStyleSheet("color: #9ece6a; font-size: 12px; padding-left: 4px;")
            self.status_lbl.setText("Đã nạp model thành công.")
        except Exception as e:
            self.model_info_lbl.setText(f"✖ Lỗi nạp model: {e}")
            self.model_info_lbl.setStyleSheet("color: #f7768e; font-size: 12px; padding-left: 4px;")
            if not silent:
                QMessageBox.critical(self, "Lỗi nạp Model", f"Không thể nạp file checkpoint:\n{e}")

    def _start_analysis(self):
        video_path = self.video_edit.text().strip()
        ckpt_path = self.ckpt_edit.text().strip()

        if not ckpt_path or not os.path.isfile(ckpt_path):
            QMessageBox.critical(self, "Chưa có Model", "Vui lòng chọn file checkpoint (.pth / .pth.tar) hợp lệ.")
            return

        if not video_path or not os.path.isfile(video_path):
            QMessageBox.warning(self, "Chưa chọn Video", "Vui lòng chọn một file video hợp lệ để kiểm tra.")
            return

        # Nạp lại model nếu đường dẫn checkpoint thay đổi
        if self.loaded_model is None or self.loaded_ckpt_path != ckpt_path:
            self._manual_load_model(silent=False)
            if self.loaded_model is None:
                return

        threshold = self.threshold_slider.value() / 100.0
        max_chunks = self.chunks_spin.value()
        max_seconds = float(self.seconds_spin.value())
        crop_mode = self.crop_combo.currentData()

        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.result_badge.setText("ĐANG XỬ LÝ...")
        self.result_badge.setStyleSheet(
            "background-color: #e0af68; color: #1a1b26; border-radius: 8px; font-size: 15px; font-weight: bold;"
        )
        self.summary_detail.setText("Đang tách frame 30fps + audio 48kHz, lọc chuyển cảnh và crop vùng miệng...")

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.worker = VideoAnalysisWorker(
            video_path=video_path,
            ckpt_path=ckpt_path,
            model=self.loaded_model,
            device=self.loaded_device,
            threshold=threshold,
            max_chunks=max_chunks,
            max_seconds=max_seconds,
            crop_mode=crop_mode,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _stop_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.status_lbl.setText("Đang dừng phân tích...")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_progress(self, cur: int, total: int, msg: str):
        if total > 0:
            pct = int((cur / total) * 100)
            self.progress_bar.setValue(min(pct, 100))
        self.status_lbl.setText(msg)

    def _on_finished(self, res: Dict[str, Any]):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_lbl.setText("Phân tích hoàn tất!")
        self.last_result = res
        self._render_result(res)

    def _render_result(self, res: Dict[str, Any]):
        overall_label = res['overall_label']
        overall_score = res['overall_score']
        overall_dist = res.get('overall_distance')
        threshold = res['threshold']

        if overall_label == 'FAKE':
            self.result_badge.setText("GIẢ MẠO\n(FAKE)")
            self.result_badge.setStyleSheet(
                "background-color: #f7768e; color: #1a1b26; border-radius: 8px; font-size: 17px; font-weight: bold;"
            )
        elif overall_label == 'REAL':
            self.result_badge.setText("VIDEO THẬT\n(REAL)")
            self.result_badge.setStyleSheet(
                "background-color: #9ece6a; color: #1a1b26; border-radius: 8px; font-size: 17px; font-weight: bold;"
            )
        else:
            self.result_badge.setText("KHÔNG ĐỦ Đ/K")
            self.result_badge.setStyleSheet(
                "background-color: #565f89; color: #ffffff; border-radius: 8px; font-size: 14px; font-weight: bold;"
            )

        v_info = res['video_info']
        score_text = f"{overall_score:.4f} ({overall_score * 100:.2f}%)" if overall_score is not None else "N/A"
        dist_text = f"{overall_dist:.4f}" if overall_dist is not None else "N/A"
        skip_summary = ", ".join(f"{k}: {v}" for k, v in res['reason_counts'].items()) or "Không có"
        lm_stats = res.get('lm_stats', {})
        lm_str = f"MediaPipe môi: {lm_stats.get('lm_ok', 0)} frame | YuNet miệng: {lm_stats.get('lm_fallback', 0)} frame"

        detail_text = (
            f"<b>Điểm xác suất FAKE trung bình:</b> "
            f"<span style='color: #7dcfff; font-size: 15px;'>{score_text}</span> "
            f"(Ngưỡng: {threshold:.2f} | Khoảng cách A-V TB: {dist_text})<br>"
            f"<b>Đoạn 1s hợp lệ:</b> {res['valid_chunks_count']} | "
            f"<b>Đoạn bị loại:</b> {res['skipped_count']} ({skip_summary})<br>"
            f"<b>Thông tin Video:</b> {v_info['width']}x{v_info['height']} @ {v_info['fps']:.1f} fps, "
            f"thời lượng {v_info['duration']:.1f}s | <b>Chế độ Crop:</b> {res.get('crop_mode')} ({lm_str})"
        )
        self.summary_detail.setText(detail_text)

        # Điền dữ liệu vào bảng
        chunks = res['chunks']
        skipped = res['skipped_chunks']
        all_entries = []
        for idx_c, c in enumerate(chunks):
            all_entries.append((
                c['sec'], c['score'], c.get('distance'), c['label'],
                f"Hợp lệ (Bhattacharyya={c.get('cut_score', 0):.2f})", idx_c
            ))
        for s in skipped:
            all_entries.append((
                s['sec'], None, None, "N/A",
                f"Bị loại: {s['reason_text']} (cut={s.get('cut_score', 0):.2f})", None
            ))

        all_entries.sort(key=lambda x: x[0])

        self.table.blockSignals(True)
        self.table.setRowCount(len(all_entries))
        first_valid_row = None

        for row, (sec, sc, dist, lab, note, chunk_idx) in enumerate(all_entries):
            sec_item = QTableWidgetItem(f"Giây {sec:02d} - {sec + 1:02d}s")
            sec_item.setTextAlignment(Qt.AlignCenter)
            sec_item.setData(Qt.UserRole, chunk_idx)
            self.table.setItem(row, 0, sec_item)

            score_str = f"{sc:.4f} ({sc * 100:.1f}%)" if sc is not None else "-"
            score_item = QTableWidgetItem(score_str)
            score_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 1, score_item)

            dist_str = f"{dist:.4f}" if dist is not None else "-"
            dist_item = QTableWidgetItem(dist_str)
            dist_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 2, dist_item)

            lab_item = QTableWidgetItem(lab)
            lab_item.setTextAlignment(Qt.AlignCenter)
            if lab == 'FAKE':
                lab_item.setForeground(QColor("#f7768e"))
            elif lab == 'REAL':
                lab_item.setForeground(QColor("#9ece6a"))
            else:
                lab_item.setForeground(QColor("#565f89"))
            self.table.setItem(row, 3, lab_item)

            note_item = QTableWidgetItem(note)
            self.table.setItem(row, 4, note_item)

            if chunk_idx is not None and first_valid_row is None:
                first_valid_row = row

        self.table.blockSignals(False)

        if first_valid_row is not None:
            self.table.selectRow(first_valid_row)
        else:
            self._clear_preview()

    def _clear_preview(self):
        self.preview_title_lbl.setText("Không có ảnh crop cho đoạn này.")
        self.overlay_img_lbl.setPixmap(QPixmap())
        self.overlay_img_lbl.setText("Không có khung hình hợp lệ")
        for idx_f, lbl in zip((0, 10, 20, 29), self.crop_labels):
            lbl.setPixmap(QPixmap())
            lbl.setText(f"F.{idx_f}")

    def _on_table_row_selected(self):
        if self.last_result is None:
            return
        selected = self.table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        first_item = self.table.item(row, 0)
        if first_item is None:
            return
        chunk_idx = first_item.data(Qt.UserRole)
        if chunk_idx is None:
            self._clear_preview()
            return

        chunks = self.last_result.get('chunks', [])
        if not (0 <= chunk_idx < len(chunks)):
            return
        c = chunks[chunk_idx]
        sec = c['sec']
        self.preview_title_lbl.setText(
            f"Giây {sec}s - {sec + 1}s | P(FAKE) = {c['score']:.4f} => {c['label']}"
        )

        ov = c.get('overlay_frame')
        if ov is not None:
            w = max(260, self.overlay_img_lbl.width() - 8)
            h = max(155, self.overlay_img_lbl.height() - 8)
            self.overlay_img_lbl.setPixmap(bgr_to_pixmap(ov, w, h))

        preview_crops = c.get('preview_crops', [])
        for i, lbl in enumerate(self.crop_labels):
            if i < len(preview_crops):
                lbl.setPixmap(bgr_to_pixmap(preview_crops[i], 92, 92))
            else:
                lbl.setPixmap(QPixmap())

    def _on_error(self, err_msg: str):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_lbl.setText("Xảy ra lỗi trong quá trình phân tích.")
        self.result_badge.setText("LỖI")
        self.result_badge.setStyleSheet(
            "background-color: #f7768e; color: #1a1b26; border-radius: 8px; font-size: 16px; font-weight: bold;"
        )
        QMessageBox.critical(self, "Lỗi phân tích Video", f"Đã xảy ra lỗi:\n{err_msg}")

