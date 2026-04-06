import os, cv2, json
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

class AdvancedTrackingPopup(QDialog):
    def __init__(self, workspace, video_list, models_list, roi_data):
        super().__init__()
        self.setWindowTitle("TAAM | Advanced Arena Tracking & Quantification")
        self.resize(1100, 850)
        self.workspace = workspace
        self.video_list = video_list
        self.models_list = models_list
        self.roi_data = roi_data # Passed from the ROI tab
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Scroller for Settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.form = QVBoxLayout(scroll_content)
        
        # 1. Video & Model Selection
        input_grp = QGroupBox("1. Batch Input Selection")
        il = QVBoxLayout()
        self.vid_selector = QListWidget()
        self.vid_selector.addItems([os.path.basename(v) for v in self.video_list])
        self.vid_selector.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.vid_selector.selectAll()
        
        il.addWidget(QLabel("Select Videos for Batch Tracking:"))
        il.addWidget(self.vid_selector)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.models_list)
        il.addWidget(QLabel("Select Trained Model:"))
        il.addWidget(self.model_combo)
        input_grp.setLayout(il); self.form.addWidget(input_grp)

        # 2. Tracking Method & Parameters
        track_grp = QGroupBox("2. Tracking Methodology")
        tl = QFormLayout()
        self.combo_method = QComboBox()
        self.combo_method.addItems(["Confidence Filter", "Custom Force-N", "Norfair", "BoTSORT"])
        
        self.spin_conf = QDoubleSpinBox(); self.spin_conf.setRange(0.01, 1.0); self.spin_conf.setValue(0.25)
        self.spin_iou = QDoubleSpinBox(); self.spin_iou.setRange(0.01, 1.0); self.spin_iou.setValue(0.45)
        self.spin_max_per_tank = QSpinBox(); self.spin_max_per_tank.setRange(1, 100); self.spin_max_per_tank.setValue(1)
        
        tl.addRow("Tracking Method:", self.combo_method)
        tl.addRow("Detection Confidence:", self.spin_conf)
        tl.addRow("IOU Match Thresh:", self.spin_iou)
        tl.addRow("Max Animals per ROI:", self.spin_max_per_tank)
        track_grp.setLayout(tl); self.form.addWidget(track_grp)

        # 3. Arena Quantification
        arena_grp = QGroupBox("3. Arena Mapping (Left-to-Right, Top-to-Bottom)")
        al = QVBoxLayout()
        self.lbl_roi_count = QLabel(f"Detected ROIs from Designer: {len(self.roi_data)}")
        self.chk_auto_assign = QCheckBox("Assign Tank ID based on Spatial Position")
        self.chk_auto_assign.setChecked(True)
        al.addWidget(self.lbl_roi_count)
        al.addWidget(self.chk_auto_assign)
        arena_grp.setLayout(al); self.form.addWidget(arena_grp)

        # 4. Export Options
        exp_grp = QGroupBox("4. Export Settings")
        el = QGridLayout()
        self.chk_vid = QCheckBox("Save Annotated Video"); self.chk_vid.setChecked(True)
        self.chk_csv = QCheckBox("Export Global CSV"); self.chk_csv.setChecked(True)
        self.chk_tank_excel = QCheckBox("Export Tank-wise Excel"); self.chk_tank_excel.setChecked(True)
        self.chk_heatmap = QCheckBox("Generate Heatmaps")
        el.addWidget(self.chk_vid, 0, 0); el.addWidget(self.chk_csv, 0, 1)
        el.addWidget(self.chk_tank_excel, 1, 0); el.addWidget(self.chk_heatmap, 1, 1)
        exp_grp.setLayout(el); self.form.addWidget(exp_grp)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # 5. Execution Console
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background: #000; color: #00ff41; font-family: Consolas;")
        self.pbar = QProgressBar()
        main_layout.addWidget(self.log_output)
        main_layout.addWidget(self.pbar)

        # Action Buttons
        btn_lay = QHBoxLayout()
        self.btn_start = QPushButton("🚀 START BATCH TRACKING")
        self.btn_start.setFixedHeight(50)
        self.btn_start.clicked.connect(self.run_process)
        self.btn_cancel = QPushButton("🛑 CANCEL")
        self.btn_cancel.setFixedHeight(50)
        btn_lay.addWidget(self.btn_start, 2)
        btn_lay.addWidget(self.btn_cancel, 1)
        main_layout.addLayout(btn_lay)

    def run_process(self):
        # Logic to extract settings and launch worker
        selected_vids = [item.text() for item in self.vid_selector.selectedItems()]
        self.log_output.append(f"INIT: Starting analysis for {len(selected_vids)} videos...")
        # (Worker implementation follows in next step)