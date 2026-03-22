import os, cv2, csv, torch, gc, numpy as np, pandas as pd, random
from PyQt6.QtCore import QThread, pyqtSignal, QRectF, QPointF
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

class ArenaWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)

    def __init__(self, config, workspace):
        super().__init__()
        self.config = config
        self.workspace = workspace
        self.is_running = True
        # Professional High-Contrast Palette
        self.class_palette = [(57, 255, 20), (255, 0, 127), (0, 255, 255), (255, 102, 0), (204, 0, 255), (0, 102, 255), (255, 255, 255)]

    def stop(self):
        self.is_running = False

    def _get_grid_cells(self, roi):
        """
        Decomposes a Grid ROI into individual sub-arenas.
        Ordering: Left-to-Right, then Top-to-Bottom.
        """
        cells = []
        rx, ry, rw, rh = roi['x'], roi['y'], roi['w'], roi['h']
        rows, cols = roi['grid']
        cw, ch = rw / cols, rh / rows

        for r in range(rows):
            for c in range(cols):
                cells.append({
                    'type': 'rect',
                    'x': rx + c * cw,
                    'y': ry + r * ch,
                    'w': cw,
                    'h': ch,
                    'is_subcell': True
                })
        return cells

    def run(self):
        try:
            self.log_signal.emit("🚀 AI: Initializing Arena-Wise Tracking Engine...")
            model_path = os.path.join(self.workspace, "Models", self.config['model_name'], "weights", "best.pt")
            model = YOLO(model_path)
            class_names = model.names

            # 1. EXPAND ARENAS PRESERVING DRAWING ORDER
            all_arenas = []
            for s in self.config['rois']:
                if s['type'] == 'grid':
                    # Grids are expanded into sub-cells (Left-to-Right, Top-to-Bottom)
                    all_arenas.extend(self._get_grid_cells(s))
                else:
                    # Rects and Circles are kept exactly as drawn
                    all_arenas.append(s)
            
            self.log_signal.emit(f"📐 GEOMETRY: {len(all_arenas)} logical tanks mapped. Logic: Drawing-order + Grid-expansion.")

            for v_idx, v_path in enumerate(self.config['videos']):
                if not self.is_running: break
                
                base_n = os.path.splitext(os.path.basename(v_path))[0]
                self.log_signal.emit(f"📽️ PROCESSING: {base_n}")
                
                cap = cv2.VideoCapture(v_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w_vid, h_vid = int(cap.get(3)), int(cap.get(4))
                
                out_dir = os.path.join(self.workspace, "Advanced_Results", base_n)
                os.makedirs(out_dir, exist_ok=True)
                
                writer = None
                if self.config['save_video']:
                    writer = cv2.VideoWriter(os.path.join(out_dir, f"{base_n}_tracked.mp4"), 
                                            cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_vid, h_vid))

                tracks = {i: {} for i in range(len(all_arenas))}
                raw_data, bg_frame = [], None

                for f_idx in range(total_f):
                    if not self.is_running: break
                    ret, frame = cap.read()
                    if not ret: break
                    if f_idx == 0: bg_frame = frame.copy()

                    res = model.predict(frame, conf=self.config['conf'], verbose=False)[0]
                    dets_by_arena = [[] for _ in range(len(all_arenas))]
                    is_seg = self.config['task_type'] == "Segmentation" and res.masks is not None

                    for i in range(len(res.boxes)):
                        box = res.boxes[i]
                        b = box.xyxy[0].cpu().numpy()
                        
                        # COORDINATE CALCULATION
                        if is_seg:
                            poly = res.masks[i].xy[0].astype(np.int32)
                            M = cv2.moments(poly)
                            cx, cy = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"] != 0 else ((b[0]+b[2])/2, (b[1]+b[3])/2)
                        else:
                            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                            poly = None

                        # SPATIAL ASSIGNMENT (Centroid must fall inside ROI)
                        for a_idx, arena in enumerate(all_arenas):
                            ax, ay, aw, ah = arena['x'], arena['y'], arena['w'], arena['h']
                            if ax <= cx <= ax + aw and ay <= cy <= ay + ah:
                                dets_by_arena[a_idx].append({
                                    'pos': (cx, cy), 'conf': float(box.conf[0]), 
                                    'box': b, 'cid': int(box.cls[0]), 'poly': poly
                                })
                                break

                    # LOCAL TRACKING PER ARENA
                    for a_idx, dets in enumerate(dets_by_arena):
                        dets.sort(key=lambda x: x['conf'], reverse=True)
                        dets = dets[:self.config['max_n']] # FORCE N
                        
                        if not tracks[a_idx] or len(tracks[a_idx]) < len(dets):
                            tracks[a_idx] = {i: d['pos'] for i, d in enumerate(dets)}
                            matches = [(i, i) for i in range(len(dets))]
                        else:
                            t_ids = list(tracks[a_idx].keys())
                            cost = np.array([[np.hypot(tracks[a_idx][tid][0]-d['pos'][0], tracks[a_idx][tid][1]-d['pos'][1]) for d in dets] for tid in t_ids])
                            ri, ci = linear_sum_assignment(cost)
                            matches = list(zip(ri, ci))

                        for r_m, c_m in matches:
                            tid, d = r_m, dets[c_m]
                            tracks[a_idx][tid] = d['pos']
                            cx, cy = d['pos']
                            raw_data.append([f_idx, a_idx+1, tid+1, class_names[d['cid']], cx, cy, d['cid']])
                            
                            if writer:
                                color = self.class_palette[d['cid'] % len(self.class_palette)]
                                if is_seg and d['poly'] is not None:
                                    # Draw semi-transparent mask
                                    mask_overlay = frame.copy()
                                    cv2.fillPoly(mask_overlay, [d['poly']], color)
                                    cv2.addWeighted(mask_overlay, 0.4, frame, 0.6, 0, frame)
                                
                                x1, y1, x2, y2 = d['box'].astype(int)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1) # RED CENTROID
                                
                                # Label: Arena Index + Object ID
                                label = f"A{a_idx+1} ID{tid+1} | {class_names[d['cid']]}"
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
                                cv2.rectangle(frame, (x1, y1-30), (x1+tw+10, y1), color, -1)
                                cv2.putText(frame, label, (x1+5, y1-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)

                    if writer: writer.write(frame)
                    if f_idx % 200 == 0:
                        self.progress_signal.emit(int((f_idx/total_f)*100), f"Vid {v_idx+1}: {int((f_idx/total_f)*100)}%")

                if writer: writer.release()
                cap.release()
                if self.is_running and len(raw_data) > 0:
                    self.log_signal.emit(f"📊 ANALYTICS: Compiling data for {base_n}...")
                    self._generate_analytics(base_n, raw_data, bg_frame, out_dir)

            self.finished_signal.emit("✅ Advanced Pipeline Completed.")
        except Exception as e:
            self.finished_signal.emit(f"❌ Error: {str(e)}")

    def _generate_analytics(self, base_n, data, bg, out):
        df = pd.DataFrame(data, columns=["Frame", "Arena", "ID", "Class", "X", "Y", "CID"])
        df.to_csv(os.path.join(out, f"{base_n}_data.csv"), index=False)
        
        if self.config.get('save_traj') and bg is not None:
            canvas = bg.copy()
            for a_id in df['Arena'].unique():
                for o_id in df[df['Arena']==a_id]['ID'].unique():
                    subset = df[(df['Arena']==a_id) & (df['ID']==o_id)]
                    pts = subset[['X', 'Y']].values.astype(np.int32)
                    cid = int(subset['CID'].iloc[0])
                    color = self.class_palette[cid % len(self.class_palette)]
                    if len(pts) > 1:
                        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out, f"{base_n}_trajectories.png"), canvas)

        if self.config.get('save_heat') and bg is not None:
            h, w = bg.shape[:2]
            accum = np.zeros((h, w), dtype=np.float32)
            for _, r in df.iterrows():
                cv2.circle(accum, (int(r['X']), int(r['Y'])), 15, 1.0, -1)
            accum = cv2.normalize(cv2.GaussianBlur(accum, (51, 51), 0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            heatmap = cv2.applyColorMap(accum, cv2.COLORMAP_JET)
            res = cv2.addWeighted(bg, 0.6, heatmap, 0.4, 0)
            cv2.imwrite(os.path.join(out, f"{base_n}_heatmap.png"), res)

        if self.config.get('save_xlsx'):
            with pd.ExcelWriter(os.path.join(out, f"{base_n}_analysis.xlsx")) as writer:
                df.to_excel(writer, sheet_name="Master_Data", index=False)
                for a_id in sorted(df['Arena'].unique()):
                    df[df['Arena']==a_id].to_excel(writer, sheet_name=f"Arena_{int(a_id)}", index=False)