import os
import sys

# CRITICAL: Fix for CuBLAS deterministic warnings & memory fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PyQt6.QtWidgets import QApplication
from gui.main_window import TAAMMainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TAAMMainWindow()
    window.show()
    sys.exit(app.exec())