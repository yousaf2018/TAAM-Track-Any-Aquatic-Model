import os, sys, cv2, glob, shutil
from PyQt6.QtWidgets import *
from PyQt6.QtCore import Qt, QSize
from ui.video_selector import VideoSelectorWidget
from workers.pipeline_worker import TAAMWorker
from workers.splitter_worker import SplitterWorker
from utils.stream_logger import StreamRedirector

class TAAMMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TAAM | Track Any Aquatic Model")
        self.resize(1650, 950); self.workspace = os.path.abspath(os.path.join(os.getcwd(), "TAAM_Workspace"))
        os.makedirs(self.workspace, exist_ok=True); self.project_annotations = {}; self.video_data = {}
        
        # REMOVED invalid CSS cursors, now handled via Python setCursor
        self.setStyleSheet(self.get_style())
        
        self.stdout_redirect = StreamRedirector(sys.stdout); self.stderr_redirect = StreamRedirector(sys.stderr)
        sys.stdout = self.stdout_redirect; sys.stderr = self.stderr_redirect
        self.setup_ui()
        self.stdout_redirect.text_written.connect(self.log_sam.append); self.stderr_redirect.text_written.connect(self.log_sam.append)
        self.refresh_model_list()

    def get_style(self):
        return """
            QMainWindow { background-color: #050505; }
            QWidget { background-color: #050505; color: #efefef; font-family: 'Segoe UI'; }
            QFrame#Sidebar { background-color: #0d0d0d; border-right: 1px solid #333; }
            QPushButton { background-color: #0078d4; border: none; padding: 10px; font-weight: bold; border-radius: 4px; color: white; }
            QPushButton:hover { background-color: #008af0; }
            QLineEdit, QSpinBox, QComboBox { background: #1a1a1a; border: 1px solid #333; padding: 5px; color: #00ff41; }
            #LogApp { color: #00ff41; } #LogSAM { color: #4db8ff; }
            QLabel#StatVal { color: #39FF14; font-weight: bold; font-size: 18px; }
        """

    def _btn(self, text, func):
        btn = QPushButton(text); btn.setCursor(Qt.CursorShape.PointingHandCursor); btn.clicked.connect(func); return btn

    def setup_ui(self):
        central = QWidget(); self.setCentralWidget(central); layout = QHBoxLayout(central); layout.setContentsMargins(0,0,0,0)
        sidebar = QFrame(); sidebar.setObjectName("Sidebar"); sidebar.setFixedWidth(360); side_lay = QVBoxLayout(sidebar)
        side_lay.addWidget(QLabel("🐬 TAAM CONTROL")); side_lay.addWidget(self._btn("📂 Workspace", self.set_workspace))
        self.lbl_ws = QLabel(self.workspace); self.lbl_ws.setStyleSheet("font-size:10px; color:#00ff41; background:#111; padding:5px;"); side_lay.addWidget(self.lbl_ws)
        
        self.list_vids = QListWidget(); self.list_vids.setCursor(Qt.CursorShape.PointingHandCursor); self.list_vids.currentRowChanged.connect(self.load_preview); side_lay.addWidget(self.list_vids)
        btn_box = QHBoxLayout(); btn_box.addWidget(self._btn("📥 Add", self.add_videos)); btn_box.addWidget(self._btn("🗑️ Rem", self.remove_video)); side_lay.addLayout(btn_box)
        
        model_group = QGroupBox("Model Manager"); m_lay = QVBoxLayout(); self.list_models = QListWidget(); self.list_models.setCursor(Qt.CursorShape.PointingHandCursor); m_lay.addWidget(self.list_models)
        m_lay.addWidget(self._btn("📂 Import External .pt", self.import_custom_model)); m_lay.addWidget(self._btn("❌ Delete Selected", self.delete_model)); model_group.setLayout(m_lay); side_lay.addWidget(model_group)
        
        stat_group = QGroupBox("Stats"); sl = QFormLayout(); self.lbl_f = QLabel("0"); self.lbl_f.setObjectName("StatVal"); self.lbl_b = QLabel("0"); self.lbl_b.setObjectName("StatVal"); sl.addRow("Annotated Vids:", self.lbl_f); sl.addRow("Total Bboxes:", self.lbl_b); stat_group.setLayout(sl); side_lay.addWidget(stat_group); side_lay.addStretch(); layout.addWidget(sidebar)

        content = QVBoxLayout(); content.setContentsMargins(15,15,15,15); self.tabs = QTabWidget()
        t0 = QWidget(); l0 = QVBoxLayout(t0); slay = QFormLayout(); self.spin_split_sec = QSpinBox(); self.spin_split_sec.setValue(60); slay.addRow("Split Seconds:", self.spin_split_sec); l0.addLayout(slay); self.list_split = QListWidget(); l0.addWidget(self.list_split); l0.addWidget(self._btn("📥 Add Huge Videos", self.add_split_videos)); l0.addWidget(self._btn("✂️ START SPLITTER", self.run_video_splitter)); self.tabs.addTab(t0, "1. PRE-PROCESS")
        t1 = QWidget(); l1 = QVBoxLayout(t1); self.video_ui = VideoSelectorWidget(); self.video_ui.selection_changed.connect(self.on_ann_change); self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setCursor(Qt.CursorShape.PointingHandCursor); self.slider.valueChanged.connect(self.seek); l1.addWidget(self.video_ui, 1); l1.addWidget(self.slider); self.tabs.addTab(t1, "2. ANNOTATE")
        
        t2 = QWidget(); l2 = QVBoxLayout(t2); self.edit_name = QLineEdit("Exp_1"); l2.addWidget(QLabel("Experiment Name:")); l2.addWidget(self.edit_name); cfg_box = QGroupBox("YOLO Parameters"); cfg_lay = QFormLayout()
        self.combo_task = QComboBox(); self.combo_task.addItems(["Detection", "Segmentation"]); self.edit_weights = QLineEdit(""); self.edit_weights.setPlaceholderText("Leave blank for default YOLOv8"); btn_browse_pt = self._btn("📁 Browse .pt", self.browse_custom_pt)
        self.spin_epochs = QSpinBox(); self.spin_epochs.setValue(25); self.spin_batch_yolo = QSpinBox(); self.spin_batch_yolo.setValue(16); self.spin_imgsz = QSpinBox(); self.spin_imgsz.setValue(640); self.spin_max = QSpinBox(); self.spin_max.setValue(500)
        cfg_lay.addRow("Task:", self.combo_task); cfg_lay.addRow("Base Weights:", self.edit_weights); cfg_lay.addRow("", btn_browse_pt); cfg_lay.addRow("Epochs:", self.spin_epochs); cfg_lay.addRow("Batch:", self.spin_batch_yolo); cfg_lay.addRow("Img Size:", self.spin_imgsz); cfg_lay.addRow("Max Frames:", self.spin_max); cfg_box.setLayout(cfg_lay); l2.addWidget(cfg_box); self.btn_run = self._btn("🚀 LAUNCH PIPELINE", self.start_pipeline); self.btn_run.setFixedHeight(70); l2.addWidget(self.btn_run); l2.addStretch(); self.tabs.addTab(t2, "3. TRAIN")
        t3 = QWidget(); l3 = QVBoxLayout(t3); l3.addWidget(self._btn("🎯 START TRACKING", self.start_tracking)); l3.addStretch(); self.tabs.addTab(t3, "4. DEPLOY")
        
        content.addWidget(self.tabs, 3); self.prog_bar = QProgressBar(); content.addWidget(self.prog_bar); log_lay = QHBoxLayout(); g1 = QGroupBox("Events"); v1 = QVBoxLayout(); self.log_app = QTextEdit(); self.log_app.setObjectName("LogApp"); v1.addWidget(self.log_app); g1.setLayout(v1); g2 = QGroupBox("Engine"); v2 = QVBoxLayout(); self.log_sam = QTextEdit(); self.log_sam.setObjectName("LogSAM"); v2.addWidget(self.log_sam); g2.setLayout(v2); log_lay.addWidget(g1); log_lay.addWidget(g2); content.addLayout(log_lay, 1); layout.addLayout(content)

    def browse_custom_pt(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select YOLO Weight File", "", "Weights (*.pt)")
        if p: self.edit_weights.setText(p)

    def set_workspace(self):
        d = QFileDialog.getExistingDirectory(self, "Workspace"); 
        if d: self.workspace = os.path.abspath(d); self.lbl_ws.setText(self.workspace); self.refresh_model_list()

    def refresh_model_list(self):
        self.list_models.clear()
        for p in glob.glob(os.path.join(self.workspace, "Models", "*/weights/best.pt")): self.list_models.addItem(p.split(os.sep)[-3])

    def import_custom_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Import YOLO Model", "", "YOLO (*.pt)")
        if p:
            dest = os.path.join(self.workspace, "Models", os.path.basename(p).replace(".pt", ""), "weights")
            os.makedirs(dest, exist_ok=True); shutil.copy(p, os.path.join(dest, "best.pt")); self.refresh_model_list()

    def delete_model(self):
        item = self.list_models.currentItem()
        if item: shutil.rmtree(os.path.join(self.workspace, "Models", item.text()), ignore_errors=True); self.refresh_model_list()

    def add_videos(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Add Videos", "", "Videos (*.mp4 *.avi)")
        for p in ps:
            if p not in self.video_data: cap = cv2.VideoCapture(p); self.video_data[p] = int(cap.get(7)); cap.release(); self.list_vids.addItem(p)

    def add_split_videos(self):
        ps, _ = QFileDialog.getOpenFileNames(self, "Add Huge Videos", "", "Videos (*.mp4 *.avi)")
        for p in ps: self.list_split.addItem(p)

    def run_video_splitter(self):
        vids = [self.list_split.item(i).text() for i in range(self.list_split.count())]
        self.sworker = SplitterWorker(vids, self.spin_split_sec.value(), self.workspace); self.sworker.log_signal.connect(self.log_app.append); self.sworker.start()

    def remove_video(self):
        idx = self.list_vids.currentRow(); 
        if idx >= 0: p = self.list_vids.item(idx).text(); self.project_annotations.pop(p, None); self.list_vids.takeItem(idx)

    def on_ann_change(self):
        if hasattr(self, 'current_video'):
            self.project_annotations[self.current_video] = self.video_ui.annotations
            self.lbl_f.setText(str(len(self.project_annotations))); self.lbl_b.setText(str(sum(sum(len(b) for b in v.values()) for v in self.project_annotations.values())))

    def load_preview(self, row):
        if row < 0: return
        self.current_video = self.list_vids.item(row).text(); cap = cv2.VideoCapture(self.current_video); total = int(cap.get(7)); self.slider.setRange(0, total-1)
        self.video_ui.annotations = self.project_annotations.get(self.current_video, {})
        ret, f = cap.read(); self.video_ui.set_current_frame(0, f); cap.release(); self.on_ann_change()

    def seek(self, val):
        if not hasattr(self, 'current_video'): return
        cap = cv2.VideoCapture(self.current_video); cap.set(1, val); ret, f = cap.read(); self.video_ui.set_current_frame(val, f); cap.release()

    def start_pipeline(self):
        vids = [self.list_vids.item(i).text() for i in range(self.list_vids.count())]
        missing = [os.path.basename(v) for v in vids if not any(len(b)>0 for b in self.project_annotations.get(v, {}).values())]
        if missing: QMessageBox.critical(self, "Audit Fail", f"Unannotated videos:\n" + "\n".join(missing)); return
        config = {"task_type": self.combo_task.currentText(), "custom_weights": self.edit_weights.text(), "epochs": self.spin_epochs.value(), "yolo_batch": self.spin_batch_yolo.value(), "imgsz": self.spin_imgsz.value(), "max_frames": self.spin_max.value(), "tr": 70, "va": 20, "chunk_duration": 5}
        self.btn_run.setEnabled(False); self.log_app.clear(); self.log_sam.clear()
        self.worker = TAAMWorker(vids, self.project_annotations, self.workspace, self.edit_name.text(), config)
        self.worker.log_app_signal.connect(self.log_app.append); self.worker.progress_signal.connect(lambda v, m: (self.prog_bar.setValue(v), self.log_app.append(f"STATUS: {m}")))
        self.worker.finished_signal.connect(lambda m: (self.btn_run.setEnabled(True), self.refresh_model_list(), QMessageBox.information(self, "TAAM", m))); self.worker.start()

    def start_tracking(self):
        item = self.list_models.currentItem()
        if not item: return
        w_path = os.path.join(self.workspace, "Models", item.text(), "weights", "best.pt")
        self.worker = TAAMWorker([self.current_video], {}, self.workspace, "", {}, w_path); self.worker.log_app_signal.connect(self.log_app.append); self.worker.start()

if __name__ == "__main__":
    app = QApplication(sys.argv); w = TAAMMainWindow(); w.show(); sys.exit(app.exec())