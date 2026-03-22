import os
import cv2
import traceback
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def assign_tank(cx, cy, rois):
    """Checks which ROI the point belongs to. Returns ROI ID or None."""
    if not rois: return "Arena_Main"
    for r_id, roi in rois.items():
        if roi['type'] == 'rect':
            x, y, w, h = roi['data']
            if x <= cx <= x + w and y <= cy <= y + h: return r_id
        elif roi['type'] == 'circle':
            rx, ry, r = roi['data']
            if ((cx - rx)**2 + (cy - ry)**2) <= r**2: return r_id
        elif roi['type'] == 'poly':
            pts = np.array(roi['data'], np.int32)
            if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0: return r_id
    return None

def export_heatmap_image(processed_detections, video_path, output_path, video_fps, frame_sample_rate=1):
    try:
        cap = cv2.VideoCapture(video_path)
        ret, base_image = cap.read()
        cap.release()
        if not ret: return

        video_h, video_w = base_image.shape[:2]
        heatmap_acc = np.zeros((video_h, video_w), dtype=np.float32)

        for frame_idx, dets in processed_detections.items():
            if frame_idx % frame_sample_rate != 0: continue
            for det in dets:
                cx, cy = det.get('cx'), det.get('cy')
                if cx is not None and cy is not None:
                    cv2.circle(heatmap_acc, (int(cx), int(cy)), radius=20, color=1, thickness=-1)
        
        blurred = cv2.GaussianBlur(heatmap_acc, (81, 81), 0)
        norm_heat = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
        super_imposed = cv2.addWeighted(color_heat, 0.5, base_image, 0.5, 0)
        
        cv2.imwrite(output_path, super_imposed)
    except Exception as e:
        print(traceback.format_exc())

def export_trajectory_image(processed_detections, video_path, output_path, rois):
    try:
        cap = cv2.VideoCapture(video_path)
        ret, base_image = cap.read()
        cap.release()
        if not ret: return
        
        animal_paths = defaultdict(list)
        for frame_idx, dets in processed_detections.items():
            for det in dets:
                tid = det.get('track_id')
                if tid is not None:
                    animal_paths[tid].append((int(det['cx']), int(det['cy'])))

        np.random.seed(42)
        for tid, points in animal_paths.items():
            color = tuple(np.random.randint(0, 220, 3).tolist())
            if len(points) > 1:
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(base_image, [pts], isClosed=False, color=color, thickness=2)
        
        # Draw ROIs
        for r_id, roi in rois.items():
            if roi['type'] == 'rect':
                x, y, w, h = roi['data']
                cv2.rectangle(base_image, (x, y), (x+w, y+h), (255, 255, 255), 2)
                cv2.putText(base_image, f"Arena {r_id}", (x, y-10), 0, 0.7, (255, 255, 255), 2)
            elif roi['type'] == 'circle':
                cx, cy, r = roi['data']
                cv2.circle(base_image, (cx, cy), r, (255, 255, 255), 2)
                cv2.putText(base_image, f"Arena {r_id}", (cx, cy-r-10), 0, 0.7, (255, 255, 255), 2)

        cv2.imwrite(output_path, base_image)
    except Exception as e:
        print(traceback.format_exc())

def export_to_excel_by_tank(processed_detections, output_path, rois):
    try:
        all_dets = [det for frame_dets in processed_detections.values() for det in frame_dets]
        if not all_dets: return
        
        tank_data = defaultdict(list)
        for det in all_dets:
            # Ensure ROI is evaluated
            tank_num = det.get('roi_id')
            if tank_num is None:
                tank_num = assign_tank(det['cx'], det['cy'], rois)
                det['roi_id'] = tank_num
                
            if tank_num is not None:
                tank_data[tank_num].append(det)

        if not tank_data: return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for tank_num, data in sorted(tank_data.items(), key=lambda x: str(x[0])):
                df = pd.DataFrame(data)
                # Clean up datatypes
                for col in ['conf', 'cx', 'cy']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                sheet_name = f"Arena_{tank_num}"
                df.to_excel(writer, sheet_name=sheet_name, index=False, float_format='%.4f')
    except Exception as e:
        print(traceback.format_exc())