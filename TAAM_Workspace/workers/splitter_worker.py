import os
import cv2
from PyQt6.QtCore import QThread, pyqtSignal

class SplitterWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, video_paths, split_seconds, output_dir):
        super().__init__()
        self.video_paths = video_paths
        self.split_seconds = split_seconds
        self.output_dir = output_dir
        self.stop_flag = False

    def run(self):
        try:
            self.log_signal.emit("🚀 Starting Batch Video Splitting...")
            total_vids = len(self.video_paths)
            
            for v_idx, v_path in enumerate(self.video_paths):
                if self.stop_flag: break
                
                v_name = os.path.splitext(os.path.basename(v_path))[0]
                self.log_signal.emit(f"\n📂 Processing Video {v_idx+1}/{total_vids}: {v_name}")
                
                # Use custom user directory
                out_dir = os.path.join(self.output_dir, v_name)
                os.makedirs(out_dir, exist_ok=True)

                cap = cv2.VideoCapture(v_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if fps <= 0 or total_frames <= 0:
                    self.log_signal.emit(f"❌ Error: Cannot read FPS/Frames for {v_name}. Skipping.")
                    cap.release()
                    continue

                frames_per_chunk = int(fps * self.split_seconds)
                chunk_idx = 1
                curr_frame = 0
                writer = None

                self.progress_signal.emit(int((v_idx/total_vids)*100), f"Splitting: {v_name}")
                
                while True:
                    if self.stop_flag: break
                    ret, frame = cap.read()
                    if not ret: break

                    if curr_frame % frames_per_chunk == 0:
                        if writer: 
                            writer.release()
                            self.log_signal.emit(f"   ✅ Saved: {v_name}_part_{chunk_idx-1:03d}.mp4")
                        
                        out_file = os.path.join(out_dir, f"{v_name}_part_{chunk_idx:03d}.mp4")
                        writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        chunk_idx += 1

                    writer.write(frame)
                    curr_frame += 1
                    
                    if curr_frame % (fps * 10) == 0:
                        self.log_signal.emit(f"   ... Processed {curr_frame}/{total_frames} frames")

                if writer: 
                    writer.release()
                    self.log_signal.emit(f"   ✅ Saved: {v_name}_part_{chunk_idx-1:03d}.mp4")
                cap.release()

            if self.stop_flag:
                self.log_signal.emit("🛑 Splitting process cancelled by user.")
                self.finished_signal.emit("Splitting Cancelled.")
            else:
                self.log_signal.emit("🎉 All videos successfully split!")
                self.finished_signal.emit("Success: Videos Splitted.")
                
        except Exception as e:
            self.log_signal.emit(f"❌ CRITICAL ERROR: {str(e)}")
            self.finished_signal.emit("Failed.")

    def stop(self):
        self.stop_flag = True