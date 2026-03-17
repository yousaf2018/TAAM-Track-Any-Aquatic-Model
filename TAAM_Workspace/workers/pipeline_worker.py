from PyQt6.QtCore import QThread, pyqtSignal
from backend.engine import TAAMEngine
import os

class TAAMWorker(QThread):
    log_app_signal = pyqtSignal(str)
    log_sam_signal = pyqtSignal(str) # Passes terminal logs to GUI
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)

    def __init__(self, video_paths, annotations_map, workspace, model_name, config, mode="FULL"):
        super().__init__()
        self.video_paths, self.annotations_map, self.workspace = video_paths, annotations_map, workspace
        self.model_name, self.config, self.mode = model_name, config, mode
        self.engine = None

    def run(self):
        # We pass both the APP log signal and SAM log signal down to the engine
        self.engine = TAAMEngine(self.workspace, self.config, self.log_app_signal)
        self.engine.log_sam = self.log_sam_signal # Map the terminal signal explicitly
        
        if self.mode == "FULL":
            if not self.engine.load_sam3(): return
            res = self.engine.run_full_pipeline(self.video_paths, self.annotations_map, self.model_name, self.progress_signal.emit)
            self.finished_signal.emit("✅ Pipeline Task Finished." if res else "🛑 Task Cancelled or Failed.")
        else:
            from ultralytics import YOLO
            model = YOLO(self.mode)
            model.track(source=self.video_paths[0], save=True, project=os.path.join(self.workspace, "Fast_Track"), device=0)
            self.finished_signal.emit("✅ Tracking Complete.")

    def stop(self):
        if self.engine: self.engine.stop_flag = True