import os, json
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

class AdvancedTrackingPopup(QDialog):
    def __init__(self, workspace, video_paths, model_names, rois, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TAAM | Advanced Arena Tracking Configuration")
        self.resize(1000, 850)
        self.workspace = workspace
        self.video_paths = video_paths
        self.model_names = model_names
        self.rois = rois # Expects list of dicts from main_window
        self.settings_path = os.path.join(self.workspace, "internal_tracking_settings.json")
        
        self.setStyleSheet(self.get_popup_style())
        self.setup_ui()
        self.load_persisted_settings()

    def get_popup_style(self):
        return """
            QDialog { background-color: #2b2b2b; }
            QWidget { background-color: #2b2b2b; color: #ffffff; font-family: 'Segoe UI'; font-size: 13px; }
            QGroupBox { border: 2px solid #3d3d3d; margin-top: 20px; font-weight: bold; color: #00a2ed; background-color: #333333; border-radius: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget { background-color: #1e1e1e; border: 1px solid #555; padding: 6px; border-radius: 4px; color: #39FF14; }
            QPushButton { background-color: #0078d4; border: none; padding: 12px; font-weight: bold; border-radius: 5px; color: white; }
            QPushButton:hover { background-color: #0086f0; }
        """

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # 1. Batch Selection
        g1 = QGroupBox("1. Batch Selection")
        l1 = QVBoxLayout()
        self.vid_list = QListWidget()
        for v in self.video_paths: self.vid_list.addItem(os.path.basename(v))
        self.vid_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.vid_list.selectAll()
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_names)
        l1.addWidget(QLabel("Select Videos:"))
        l1.addWidget(self.vid_list)
        l1.addWidget(QLabel("Select Model:"))
        l1.addWidget(self.model_combo)
        g1.setLayout(l1); scroll_layout.addWidget(g1)

        # 2. Methodology
        g2 = QGroupBox("2. Core Methodology")
        l2 = QFormLayout()
        self.combo_task = QComboBox(); self.combo_task.addItems(["Detection", "Segmentation"])
        self.combo_method = QComboBox(); self.combo_method.addItems(["BYTETrack (Robust)", "Custom Force-N", "Norfair"])
        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0.01, 1.0); self.spin_conf.setValue(0.25)
        self.spin_max_n = QSpinBox(); self.spin_max_n.setRange(1, 99999); self.spin_max_n.setValue(1)
        l2.addRow("Task:", self.combo_task); l2.addRow("Algorithm:", self.combo_method); l2.addRow("Conf:", self.spin_conf); l2.addRow("Max N:", self.spin_max_n)
        g2.setLayout(l2); scroll_layout.addWidget(g2)

        # 3. Tuning
        self.tune_grp = QGroupBox("3. Fine-Tuning")
        tl = QFormLayout()
        self.spin_dist = QDoubleSpinBox(); self.spin_dist.setRange(1, 9999); self.spin_dist.setValue(50)
        self.spin_memory = QSpinBox(); self.spin_memory.setRange(1, 9999); self.spin_memory.setValue(30)
        tl.addRow("Jump (px):", self.spin_dist); tl.addRow("Memory (f):", self.spin_memory)
        self.tune_grp.setLayout(tl); scroll_layout.addWidget(self.tune_grp)

        # 4. Export
        g4 = QGroupBox("4. Export Settings")
        l4 = QGridLayout()
        self.chk_vid = QCheckBox("Video Overlay"); self.chk_vid.setChecked(True)
        self.chk_traj = QCheckBox("Trajectories"); self.chk_traj.setChecked(True)
        self.chk_heat = QCheckBox("Heatmap"); self.chk_heat.setChecked(True)
        self.chk_xlsx = QCheckBox("Excel Split"); self.chk_xlsx.setChecked(True)
        l4.addWidget(self.chk_vid, 0, 0); l4.addWidget(self.chk_traj, 0, 1); l4.addWidget(self.chk_heat, 1, 0); l4.addWidget(self.chk_xlsx, 1, 1)
        g4.setLayout(l4); scroll_layout.addWidget(g4)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        self.btn_go = QPushButton("🚀 LAUNCH BATCH TRACKING")
        self.btn_go.setFixedHeight(60); self.btn_go.clicked.connect(self.save_and_accept)
        layout.addWidget(self.btn_go)

    def load_persisted_settings(self):
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r') as f:
                    s = json.load(f)
                    self.combo_task.setCurrentText(s.get("task_type", "Detection"))
                    self.combo_method.setCurrentText(s.get("method", "BYTETrack (Robust)"))
                    self.spin_conf.setValue(s.get("conf", 0.25))
                    self.spin_max_n.setValue(s.get("max_n", 1))
            except: pass

    def save_and_accept(self):
        settings = self.get_config()
        try:
            persist = {k: v for k, v in settings.items() if k not in ['videos', 'rois']}
            with open(self.settings_path, 'w') as f: json.dump(persist, f)
        except: pass
        self.accept()

    def get_config(self):
        selected_vids = [self.video_paths[i] for i in range(self.vid_list.count()) if self.vid_list.item(i).isSelected()]
        return {
            "videos": selected_vids, "model_name": self.model_combo.currentText(),
            "task_type": self.combo_task.currentText(), "method": self.combo_method.currentText(),
            "conf": self.spin_conf.value(), "max_n": self.spin_max_n.value(), "rois": self.rois,
            "dist_thresh": self.spin_dist.value(), "memory": self.spin_memory.value(),
            "save_video": self.chk_vid.isChecked(), "save_traj": self.chk_traj.isChecked(),
            "save_heat": self.chk_heat.isChecked(), "save_xlsx": self.chk_xlsx.isChecked()
        }