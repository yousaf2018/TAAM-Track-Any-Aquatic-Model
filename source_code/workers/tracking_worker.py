import os, cv2, csv, json, traceback
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from utils.exporters import export_heatmap_image, export_trajectory_image, export_to_excel_by_tank, assign_tank

class TAAMTrackerWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)

    def __init__(self, video_path, model_path, rois, output_dir, config):
        super().__init__()
        self.video_path = video_path
        self.model_path = model_path
        self.rois = rois 
        self.output_dir = output_dir
        self.config = config
        self.is_running = True

    def stop(self):
        self.is_running = False

    def run(self):
        try:
            self.log_signal.emit(f"Loading Deployment Model: {os.path.basename(self.model_path)}")
            model = YOLO(self.model_path)
            class_names = model.names
            class_colors = {i: tuple(np.random.randint(60, 255, size=3).tolist()) for i in class_names.keys()}

            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            out_root = os.path.join(self.output_dir, "Tracking_Analytics", base_name)
            os.makedirs(out_root, exist_ok=True)

            cap = cv2.VideoCapture(self.video_path)
            fps, w, h = cap.get(5), int(cap.get(3)), int(cap.get(4))
            total_frames = int(cap.get(7))
            
            out_vid_path = os.path.join(out_root, f"{base_name}_analyzed.mp4")
            writer = cv2.VideoWriter(out_vid_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            all_detections = {}
            csv_data = []

            self.log_signal.emit("Starting High-Speed Analytics Stream...")
            
            # Exact logic from EthoGrid YOLO Stream
            results_generator = model.track(source=self.video_path, stream=True, persist=True, verbose=False, device=0)
            
            for frame_idx, results in enumerate(results_generator):
                if not self.is_running: break
                frame = results.orig_img.copy()
                overlay = frame.copy()
                all_detections[frame_idx] = []

                if results.boxes is not None and results.boxes.id is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    track_ids = results.boxes.id.int().cpu().tolist()
                    clss = results.boxes.cls.int().cpu().tolist()
                    confs = results.boxes.conf.cpu().tolist()
                    masks = results.masks.xy if results.masks else [None]*len(boxes)

                    for box, tid, cls, conf, mask in zip(boxes, track_ids, clss, confs, masks):
                        x1, y1, x2, y2 = box
                        cx, cy = (x1+x2)/2, (y1+y2)/2
                        
                        # Apply ROI Filter
                        roi_id = assign_tank(cx, cy, self.rois)
                        if roi_id is None: continue 

                        c_name = class_names.get(cls, "Obj")
                        color = class_colors.get(cls, (0, 255, 0))

                        if mask is not None and len(mask) > 0:
                            pts = np.int32([mask])
                            cv2.fillPoly(overlay, pts, color)
                            poly_str = ";".join([f"{p[0]},{p[1]}" for p in mask])
                        else:
                            poly_str = ""

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"ID:{tid} {c_name}", (int(x1), int(y1)-10), 0, 0.6, color, 2)
                        cv2.circle(frame, (int(cx), int(cy)), 4, (0,0,255), -1)

                        det_dict = {'frame_idx': frame_idx, 'track_id': tid, 'class_name': c_name, 'conf': conf, 'cx': cx, 'cy': cy, 'roi_id': roi_id, 'polygon': poly_str}
                        all_detections[frame_idx].append(det_dict)
                        csv_data.append(det_dict)

                # Draw ROIs on output video
                for r_id, roi in self.rois.items():
                    if roi['type'] == 'rect':
                        rx, ry, rw, rh = roi['data']
                        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (255,255,255), 2)
                        cv2.putText(frame, f"Arena {r_id}", (rx, ry-5), 0, 0.7, (255,255,255), 2)
                    elif roi['type'] == 'circle':
                        cx, cy, r = roi['data']
                        cv2.circle(frame, (cx, cy), r, (255,255,255), 2)
                        cv2.putText(frame, f"Arena {r_id}", (cx, cy-r-5), 0, 0.7, (255,255,255), 2)
                    elif roi['type'] == 'poly':
                        pts = np.array(roi['data'], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], isClosed=True, color=(255,255,255), thickness=2)
                        cv2.putText(frame, f"Arena {r_id}", (pts[0][0][0], pts[0][0][1]-5), 0, 0.7, (255,255,255), 2)

                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                writer.write(frame)

                if frame_idx % 10 == 0:
                    self.progress_signal.emit(int((frame_idx/total_frames)*100), f"Tracking Frame {frame_idx}/{total_frames}")

            cap.release(); writer.release()
            self.log_signal.emit(f"✅ Video saved to: {out_vid_path}")

            # GENERATE QUANTITATIVE ETHOGRID REPORTS
            self.log_signal.emit("Generating Quantitative Reports...")
            
            csv_path = os.path.join(out_root, f"{base_name}_Tracking_Data.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['frame_idx', 'track_id', 'class_name', 'conf', 'cx', 'cy', 'roi_id', 'polygon'])
                writer.writeheader(); writer.writerows(csv_data)

            export_to_excel_by_tank(all_detections, os.path.join(out_root, f"{base_name}_Arena_Stats.xlsx"), self.rois)
            export_heatmap_image(all_detections, self.video_path, os.path.join(out_root, f"{base_name}_Heatmap.png"), fps)
            export_trajectory_image(all_detections, self.video_path, os.path.join(out_root, f"{base_name}_Trajectories.png"), self.rois)

            self.finished_signal.emit("✅ Full Quantitative Tracking Complete!")

        except Exception as e:
            self.log_signal.emit(f"❌ TRACKING CRITICAL ERROR: {str(e)}")
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit("Failed.")