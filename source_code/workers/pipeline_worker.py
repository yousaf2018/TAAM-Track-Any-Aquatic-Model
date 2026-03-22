import os
import sys
import time
import traceback
from PyQt6.QtCore import QThread, pyqtSignal
from backend.engine import TAAMEngine

class TAAMWorker(QThread):
    """
    The Orchestrator Thread for TAAM.
    Handles the asynchronous execution of SAM3 Tracking and YOLO Training.
    """
    log_app_signal = pyqtSignal(str)     # For the Green Project Events Panel
    log_sam_signal = pyqtSignal(str)     # For the Blue Engine Monitor Panel
    progress_signal = pyqtSignal(int, str) # percent, status_message
    finished_signal = pyqtSignal(str)    # final message

    def __init__(self, vids, anns, workspace, model_name, config, mode="FULL"):
        super().__init__()
        self.vids = vids
        self.anns = anns
        self.workspace = workspace
        self.model_name = model_name
        self.config = config
        self.mode = mode # "FULL", "SAM3_ONLY", or "YOLO_ONLY"
        self.engine = None

    def run(self):
        try:
            # Initialize the Backend Engine
            # We pass the signals so the engine can talk directly to the GUI panels
            self.engine = TAAMEngine(self.workspace, self.config, self.log_app_signal)
            
            # 1. Map the SAM Monitor signal if redirector is bypassed
            self.engine.log_sam = self.log_sam_signal 

            video_info = {}

            # ==========================================================
            # STAGE 1: SAM3 TRACKING (Generation of CSV and Image Pool)
            # ==========================================================
            if self.mode == "SAM3_ONLY" or self.mode == "FULL":
                self.log_app_signal.emit("STAGE 1: Initializing SAM3 Tracking Engine...")
                
                # Load the heavy SAM3 model into VRAM
                if not self.engine.load_sam3():
                    self.log_app_signal.emit("❌ CRITICAL: SAM3 Model failed to load.")
                    return

                # Process all videos in the queue
                # This generates the tracking data and the sampling image pool
                video_info = self.engine.run_sam3_only(
                    self.vids, 
                    self.anns, 
                    self.model_name, 
                    self.progress_signal.emit
                )

                if not video_info:
                    self.log_app_signal.emit("⚠️ Stage 1 stopped: No data was generated.")
                    self.finished_signal.emit("🛑 Pipeline Stopped.")
                    return

                if self.mode == "SAM3_ONLY":
                    self.log_app_signal.emit("✅ Stage 1 Complete: SAM3 data is ready in workspace.")
                    self.finished_signal.emit("Success: SAM3 Data Generated.")
                    return

            # ==========================================================
            # STAGE 2: YOLO TRAINING (Dataset creation and Model training)
            # ==========================================================
            if self.mode == "YOLO_ONLY" or self.mode == "FULL":
                self.log_app_signal.emit("STAGE 2: Initializing YOLO Training Engine...")

                # If we are starting from YOLO_ONLY, we need to find the data on disk
                if self.mode == "YOLO_ONLY":
                    self.log_app_signal.emit("🔍 Mode: YOLO Only. Scanning workspace for existing SAM3 output...")
                    # The engine will look into Experiments/[model_name] for CSVs
                    video_info = self.engine._discover_existing_data(self.model_name)
                
                if not video_info:
                    self.log_app_signal.emit("❌ ERROR: No SAM3 tracking data found. You must run Stage 1 first.")
                    self.finished_signal.emit("Failed: Data Missing.")
                    return

                # Execute YOLO Training
                # This function handles the VRAM purge and native training call
                success = self.engine.run_yolo_only(
                    video_info, 
                    self.model_name, 
                    self.progress_signal.emit
                )

                if success:
                    self.log_app_signal.emit("✅ Stage 2 Complete: Custom YOLO model trained.")
                    self.finished_signal.emit("Success: Pipeline Finished.")
                else:
                    self.log_app_signal.emit("❌ ERROR: YOLO Training failed or was cancelled.")
                    self.finished_signal.emit("🛑 Pipeline Error.")

        except Exception as e:
            # Catch any unexpected Python errors to prevent the GUI from hanging
            error_details = traceback.format_exc()
            print(error_details)
            self.log_app_signal.emit(f"💥 CRITICAL WORKER CRASH: {str(e)}")
            self.finished_signal.emit("❌ Crash.")

        finally:
            # Cleanup reference to allow garbage collection
            self.engine = None

    def stop(self):
        """
        Sets the stop flag in the engine. 
        The engine loops check this flag to terminate safely.
        """
        if self.engine:
            self.engine.stop_flag = True
            self.log_app_signal.emit("🛑 Stop signal sent. Finalizing current frame and exiting...")