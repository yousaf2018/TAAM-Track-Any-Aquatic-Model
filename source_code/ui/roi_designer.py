import json, os, cv2
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

class ROIDesigner(QWidget):
    roi_selected = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.pixmap = None
        self.shapes = [] 
        self.current_type = "rect"
        self.selected_idx = -1
        self.dragging = False
        self.last_mouse_pos = None
        self.start_pt = None
        self.scale_factor = 1.0
        self.img_rect = QRect()
        self.colors = [QColor(57, 255, 20, 100), QColor(255, 0, 255, 100), QColor(0, 255, 255, 100)]
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #333;")

    def set_frame(self, frame_bgr):
        if frame_bgr is None: return
        h, w, ch = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(img)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.pixmap:
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select a video to begin ROI design.")
            return

        ww, wh = self.width(), self.height()
        iw, ih = self.pixmap.width(), self.pixmap.height()
        self.scale_factor = min(ww/iw, wh/ih)
        nw, nh = int(iw*self.scale_factor), int(ih*self.scale_factor)
        self.img_rect = QRect((ww-nw)//2, (wh-nh)//2, nw, nh)
        painter.drawPixmap(self.img_rect, self.pixmap)

        for i, s in enumerate(self.shapes):
            painter.save()
            rect = s['points']
            sw, sh = rect.width() * self.scale_factor, rect.height() * self.scale_factor
            scx = self.img_rect.x() + rect.center().x() * self.scale_factor
            scy = self.img_rect.y() + rect.center().y() * self.scale_factor
            painter.translate(scx, scy)
            painter.rotate(s.get('angle', 0))
            draw_rect = QRectF(-sw/2, -sh/2, sw, sh)
            
            pen = QPen(QColor(255, 255, 255 if i == self.selected_idx else 0), 3 if i == self.selected_idx else 2)
            painter.setPen(pen)
            painter.setBrush(QBrush(self.colors[i % len(self.colors)]))

            if s['type'] == 'circle': painter.drawEllipse(draw_rect)
            else: 
                painter.drawRect(draw_rect)
                if s['type'] == 'grid':
                    r, c = s['grid']
                    for row in range(1, r):
                        y = draw_rect.top() + (draw_rect.height() * row / r)
                        painter.drawLine(QPointF(draw_rect.left(), y), QPointF(draw_rect.right(), y))
                    for col in range(1, c):
                        x = draw_rect.left() + (draw_rect.width() * col / c)
                        painter.drawLine(QPointF(x, draw_rect.top()), QPointF(x, draw_rect.bottom()))
            
            painter.setPen(Qt.GlobalColor.white)
            # Display Consistent ROI Number (1-based for user)
            painter.drawText(draw_rect.topLeft(), f" ARENA {i+1}")
            painter.restore()

        if self.start_pt and self.last_mouse_pos and not self.dragging:
            painter.setPen(QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine))
            painter.drawRect(QRect(self.start_pt, self.last_mouse_pos))

    def mousePressEvent(self, event):
        if not self.pixmap: return
        img_pt = QPointF((event.pos().x()-self.img_rect.x())/self.scale_factor, (event.pos().y()-self.img_rect.y())/self.scale_factor)
        found = -1
        for i in range(len(self.shapes)-1, -1, -1):
            if self.shapes[i]['points'].contains(img_pt):
                found = i; break
        if found != -1:
            self.selected_idx = found; self.dragging = True; self.last_mouse_pos = event.pos(); self.roi_selected.emit(found)
        else:
            self.selected_idx = -1; self.start_pt = event.pos(); self.last_mouse_pos = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_idx != -1:
            delta = (event.pos() - self.last_mouse_pos) / self.scale_factor
            self.shapes[self.selected_idx]['points'].translate(delta.x(), delta.y())
            self.last_mouse_pos = event.pos()
        elif self.start_pt: self.last_mouse_pos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        if not self.dragging and self.start_pt:
            p1, p2 = self.start_pt, event.pos()
            ix, iy = (min(p1.x(), p2.x()) - self.img_rect.x()) / self.scale_factor, (min(p1.y(), p2.y()) - self.img_rect.y()) / self.scale_factor
            iw, ih = abs(p1.x() - p2.x()) / self.scale_factor, abs(p1.y() - p2.y()) / self.scale_factor
            if iw > 10 and ih > 10:
                new_shape = {'type': self.current_type, 'points': QRectF(ix, iy, iw, ih), 'angle': 0}
                if self.current_type == "grid":
                    r, ok1 = QInputDialog.getInt(self, "Grid", "Rows:", 2, 1, 50)
                    c, ok2 = QInputDialog.getInt(self, "Grid", "Cols:", 2, 1, 50)
                    if ok1 and ok2: new_shape['grid'] = (r, c)
                self.shapes.append(new_shape)
                self.selected_idx = len(self.shapes)-1; self.roi_selected.emit(self.selected_idx)
        self.dragging = False; self.start_pt = None; self.update()

    def save_to_json(self, path):
        data = []
        for s in self.shapes:
            item = {'type': s['type'], 'x': s['points'].x(), 'y': s['points'].y(), 'w': s['points'].width(), 'h': s['points'].height(), 'angle': s.get('angle', 0)}
            if s['type'] == 'grid': item['grid'] = s['grid']
            data.append(item)
        with open(path, 'w') as f: json.dump(data, f)

    def load_from_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f); self.shapes = []
            for d in data: self.shapes.append({'type': d['type'], 'points': QRectF(d['x'], d['y'], d['w'], d['h']), 'grid': d.get('grid', (1,1)), 'angle': d.get('angle', 0)})
        self.update()