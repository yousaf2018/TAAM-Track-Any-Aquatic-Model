import sys
from PyQt6.QtCore import QObject, pyqtSignal

class StreamRedirector(QObject):
    text_written = pyqtSignal(str)
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, text):
        self.stream.write(text)
        if text.strip():
            # Clean TQDM carriage returns for clean GUI display
            clean_text = text.replace('\r', '').strip()
            if clean_text:
                self.text_written.emit(clean_text)

    def flush(self):
        self.stream.flush()