from PyQt6.QtCore import QThread, pyqtSignal
from backend.engine import TAAMEngine
import traceback
import os

class TAAMWorker(QThread):
    log_app_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)

    def __init__(self, video_paths, annotations_map, workspace, model_name, config, mode="FULL"):
        super().__init__()
        self.video_paths = video_paths
        self.annotations_map = annotations_map
        self.workspace = workspace
        self.model_name = model_name
        self.config = config
        self.mode = mode
        self.engine = None

    def run(self):
        try:
            self.engine = TAAMEngine(self.workspace, self.config, self.log_app_signal)
            if self.mode == "FULL":
                if not self.engine.load_sam3(): return
                res = self.engine.run_full_pipeline(self.video_paths, self.annotations_map, self.model_name, self.progress_signal.emit)
                self.finished_signal.emit("✅ Pipeline Task Finished." if res else "🛑 Task Cancelled or Failed.")
            else:
                from ultralytics import YOLO
                model = YOLO(self.mode)
                model.track(source=self.video_paths[0], save=True, project=os.path.join(self.workspace, "Fast_Track"), device=0)
                self.finished_signal.emit("✅ Tracking Complete.")
        except Exception as e:
            self.log_app_signal.emit(f"CRITICAL WORKER ERROR: {str(e)}")
            traceback.print_exc()
            self.finished_signal.emit("❌ Failed.")

    def stop(self):
        if self.engine: self.engine.stop_flag = True