from PyQt6.QtWidgets import QWidget, QMenu
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QAction
import cv2

class VideoSelectorWidget(QWidget):
    selection_changed = pyqtSignal() 

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus) # Required for keyboard shortcuts
        
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.img_draw_rect = QRect()
        
        self.annotations = {} # { frame_idx: [(QRect, class_id), ...] }
        self.current_frame_idx = 0
        self.selected_index = -1
        self.current_class_id = 0
        self.copy_buffer = [] # To store copied boxes
        
        self.is_drawing = False
        self.mode = "IDLE"
        self.start_pt = QPoint()
        self.drag_offset = QPoint()
        
        self.class_colors = [QColor(57, 255, 20), QColor(255, 50, 50), QColor(50, 100, 255), QColor(255, 255, 0)]

    def set_current_class(self, cid): self.current_class_id = cid

    def set_current_frame(self, frame_idx, frame_bgr):
        self.current_frame_idx = frame_idx
        if frame_bgr is not None:
            h, w, ch = frame_bgr.shape
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qt_img)
        self.selected_index = -1
        self.update()

    def get_stats(self):
        """Calculates accurate stats for the GUI."""
        annotated_frames = len(self.annotations)
        total_boxes = sum(len(v) for v in self.annotations.values())
        unique_classes = set()
        for frame_list in self.annotations.values():
            for _, cid in frame_list: unique_classes.add(cid)
        return annotated_frames, total_boxes, len(unique_classes)

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.original_pixmap: return
        
        # Calculate scaling
        ww, wh = self.width(), self.height()
        iw, ih = self.original_pixmap.width(), self.original_pixmap.height()
        self.scale_factor = min(ww/iw, wh/ih)
        nw, nh = int(iw*self.scale_factor), int(ih*self.scale_factor)
        self.img_draw_rect = QRect((ww-nw)//2, (wh-nh)//2, nw, nh)
        
        painter.drawPixmap(self.img_draw_rect, self.original_pixmap)
        
        # Draw all boxes for current frame
        cur_ann = self.annotations.get(self.current_frame_idx, [])
        for i, (rect, cid) in enumerate(cur_ann):
            color = self.class_colors[cid % len(self.class_colors)]
            pen = QPen(color, 2)
            if i == self.selected_index: 
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setWidth(3)
            painter.setPen(pen)
            
            # Map image coords to screen coords
            sx = self.img_draw_rect.x() + rect.x() * self.scale_factor
            sy = self.img_draw_rect.y() + rect.y() * self.scale_factor
            sw = rect.width() * self.scale_factor
            sh = rect.height() * self.scale_factor
            painter.drawRect(QRectF(sx, sy, sw, sh))
            painter.drawText(int(sx), int(sy-5), f"Obj:{cid}")

    def mousePressEvent(self, event):
        if not self.img_draw_rect.contains(event.pos()): return
        
        # Clicked coordinate in image space
        ix = int((event.pos().x() - self.img_draw_rect.x()) / self.scale_factor)
        iy = int((event.pos().y() - self.img_draw_rect.y()) / self.scale_factor)
        img_pt = QPoint(ix, iy)

        # 1. Check if clicking an existing box
        cur_ann = self.annotations.get(self.current_frame_idx, [])
        for i, (rect, cid) in enumerate(reversed(cur_ann)):
            real_idx = len(cur_ann) - 1 - i
            if rect.contains(img_pt):
                self.selected_index = real_idx
                self.mode = "MOVING"
                self.drag_offset = img_pt - rect.topLeft()
                self.update()
                return

        # 2. Otherwise, start drawing
        if event.button() == Qt.MouseButton.LeftButton:
            self.mode = "DRAWING"
            self.start_pt = img_pt
            self.selected_index = -1
            self.update()

    def mouseMoveEvent(self, event):
        ix = int((event.pos().x() - self.img_draw_rect.x()) / self.scale_factor)
        iy = int((event.pos().y() - self.img_draw_rect.y()) / self.scale_factor)
        img_pt = QPoint(ix, iy)

        if self.mode == "MOVING" and self.selected_index != -1:
            rect, cid = self.annotations[self.current_frame_idx][self.selected_index]
            rect.moveTo(img_pt - self.drag_offset)
            self.update()
        
    def mouseReleaseEvent(self, event):
        if self.mode == "DRAWING":
            ix = int((event.pos().x() - self.img_draw_rect.x()) / self.scale_factor)
            iy = int((event.pos().y() - self.img_draw_rect.y()) / self.scale_factor)
            new_rect = QRect(self.start_pt, QPoint(ix, iy)).normalized()
            if new_rect.width() > 5:
                if self.current_frame_idx not in self.annotations: self.annotations[self.current_frame_idx] = []
                self.annotations[self.current_frame_idx].append((new_rect, self.current_class_id))
        
        self.mode = "IDLE"
        self.selection_changed.emit()
        self.update()

    def keyPressEvent(self, event):
        # Ctrl+C Copy
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_C:
            if self.selected_index != -1:
                self.copy_buffer = [self.annotations[self.current_frame_idx][self.selected_index]]
        
        # Ctrl+V Paste
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_V:
            if self.copy_buffer:
                if self.current_frame_idx not in self.annotations: self.annotations[self.current_frame_idx] = []
                for box_data in self.copy_buffer:
                    self.annotations[self.current_frame_idx].append((QRect(box_data[0]), box_data[1]))
                self.selection_changed.emit()
                self.update()

        # Delete
        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            if self.selected_index != -1:
                del self.annotations[self.current_frame_idx][self.selected_index]
                self.selected_index = -1
                self.selection_changed.emit()
                self.update()

    def clear_current_frame(self):
        if self.current_frame_idx in self.annotations:
            del self.annotations[self.current_frame_idx]
            self.selection_changed.emit()
            self.update()