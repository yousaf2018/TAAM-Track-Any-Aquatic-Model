import os
import sys
import traceback
import multiprocessing

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.engine import TAAMEngine

class ProcessOutputRedirector:
    def __init__(self, queue, prefix="SAM_RAW:"):
        self.queue = queue
        self.prefix = prefix
    def write(self, text):
        if text.strip():
            self.queue.put(f"{self.prefix}{text.strip()}")
    def flush(self):
        pass

def run_engine_process(log_queue, video_paths, annotations_map, workspace, model_name, config):
    # Hijack stdout/stderr to stream SAM3 terminal output to the Blue Panel
    sys.stdout = ProcessOutputRedirector(log_queue, "SAM_RAW:")
    sys.stderr = ProcessOutputRedirector(log_queue, "SAM_RAW:")

    try:
        def log_func(msg): log_queue.put(f"APP_LOG:{msg}")
        def prog_func(val, msg): log_queue.put(f"PROG:{val}:{msg}")

        engine = TAAMEngine(workspace, config, log_func)
        if not engine.load_sam3():
            log_queue.put("FINISH:FAILED")
            return
        
        success = engine.run_full_pipeline(video_paths, annotations_map, model_name, prog_func)
        log_queue.put("FINISH:SUCCESS" if success else "FINISH:FAILED")
    
    except Exception as e:
        log_queue.put(f"APP_LOG:❌ CRITICAL ERROR: {str(e)}")
        log_queue.put("FINISH:FAILED")