from PyQt6.QtCore import QThread, pyqtSignal
from backend.sam_engine import SAM3Engine

class TrainingWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, video_path, annotations, project_dir, config):
        super().__init__()
        self.video_path, self.annotations, self.project_dir, self.config = \
            video_path, annotations, project_dir, config

    def run(self):
        try:
            self.log_signal.emit("Initializing SAM 3 Engine...")
            engine = SAM3Engine(self.project_dir, self.config)
            
            self.log_signal.emit("Extracting templates...")
            engine.extract_templates_from_rects(self.video_path, self.annotations)
            
            self.log_signal.emit("Processing Video...")
            # progress_callback lambda to map to GUI
            engine.process_video(self.video_path, lambda p, t, m, b, tb: self.progress_signal.emit(p, m))
            
            self.finished_signal.emit("Complete")
        except Exception as e:
            self.log_signal.emit(f"Error: {str(e)}")