import os, cv2, csv, torch, gc, numpy as np, pandas as pd, random
import time, queue, threading
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment


# ==========================================
# 🚀 THREADED VIDEO READER
# ==========================================
class ThreadedVideoReader:
    def __init__(self, video_path, batch_size=8):
        self.cap = cv2.VideoCapture(video_path)
        self.batch_size = batch_size

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(3))
        self.height = int(self.cap.get(4))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.q = queue.Queue(maxsize=batch_size * 5)
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    break
                self.q.put(frame)
            else:
                time.sleep(0.005)

    def get_batch(self):
        batch = []
        while len(batch) < self.batch_size:
            try:
                frame = self.q.get(timeout=0.05)
                batch.append(frame)
            except:
                if not self.running:
                    break
        return batch

    def release(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()


# ==========================================
# 🎯 MAIN WORKER
# ==========================================
class ArenaWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(str)

    def __init__(self, config, workspace):
        super().__init__()
        self.config = config
        self.workspace = workspace
        self.is_running = True

        self.vibrant_palette = [
            (57,255,20),(255,0,127),(0,255,255),(255,102,0),
            (204,0,255),(0,102,255),(128,255,0),(255,255,255)
        ]

    def stop(self):
        self.is_running = False

    def _get_grid_cells(self, roi):
        cells = []
        rx, ry, rw, rh = roi['x'], roi['y'], roi['w'], roi['h']
        rows, cols = roi['grid']
        cw, ch = rw / cols, rh / rows

        # Top-left → right numbering
        for r in range(rows):
            for c in range(cols):
                cells.append({
                    'type': 'rect',
                    'x': rx + c * cw,
                    'y': ry + r * ch,
                    'w': cw,
                    'h': ch
                })
        return cells

    def _log_error_to_csv(self, video, stage, err):
        path = os.path.join(self.workspace, "Advanced_Results", "results_logs.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.isfile(path)

        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["time","video","stage","error"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), video, stage, str(err)])

    # ==========================================
    # 🚀 MAIN RUN
    # ==========================================
    def run(self):
        try:
            self.log_signal.emit("🚀 Initializing Arena Engine...")

            model_path = os.path.join(
                self.workspace, "Models",
                self.config['model_name'], "weights", "best.pt"
            )
            model = YOLO(model_path)
            class_names = model.names

            # ===== GPU AUTO DETECT =====
            device_target = 'cpu'
            use_half = False

            if torch.cuda.is_available():
                self.log_signal.emit(f"🔍 GPU: {torch.cuda.get_device_name(0)}")
                try:
                    model.predict(np.zeros((160,160,3)), device=0, half=True, verbose=False)
                    device_target = 0
                    use_half = True
                    self.log_signal.emit("✅ GPU FP16 Enabled")
                except:
                    device_target = 0
                    self.log_signal.emit("✅ GPU FP32 Enabled")
            else:
                self.log_signal.emit("⚠️ No GPU detected")

            # ===== ROI =====
            all_arenas = []
            for s in self.config['rois']:
                if s['type'] == 'grid':
                    all_arenas.extend(self._get_grid_cells(s))
                else:
                    all_arenas.append(s)

            self.log_signal.emit(f"📐 {len(all_arenas)} arenas mapped")

            # ===== VIDEO LOOP =====
            for v_path in self.config['videos']:
                if not self.is_running:
                    break

                base_n = os.path.splitext(os.path.basename(v_path))[0]
                self.log_signal.emit(f"\n📽️ Processing: {base_n}")

                reader = None
                writer = None

                try:
                    reader = ThreadedVideoReader(v_path, batch_size=8)

                    if reader.total_frames == 0:
                        raise ValueError("Video has 0 frames")

                    out_dir = os.path.join(self.workspace, "Advanced_Results", base_n)
                    os.makedirs(out_dir, exist_ok=True)

                    if self.config['save_video']:
                        writer = cv2.VideoWriter(
                            os.path.join(out_dir, f"{base_n}_tracked.mp4"),
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            reader.fps,
                            (reader.width, reader.height)
                        )

                    tracks = {i: {} for i in range(len(all_arenas))}
                    raw_data = []
                    bg_frame = None

                    frame_idx = 0
                    start_time = time.time()
                    fps_timer = time.time()
                    frame_counter = 0

                    # ===== FRAME LOOP =====
                    while self.is_running:
                        batch = reader.get_batch()
                        if not batch:
                            break

                        try:
                            results = model.predict(
                                source=batch,
                                conf=self.config['conf'],
                                stream=True,
                                device=device_target,
                                half=use_half,
                                verbose=False
                            )
                        except Exception as e:
                            self._log_error_to_csv(base_n, "inference", e)
                            continue

                        for i, res in enumerate(results):
                            try:
                                frame = batch[i]
                                if frame_idx == 0:
                                    bg_frame = frame.copy()

                                dets_by_arena = [[] for _ in range(len(all_arenas))]
                                is_seg = self.config['task_type']=="Segmentation" and res.masks is not None

                                for j in range(len(res.boxes)):
                                    box = res.boxes[j]
                                    b = box.xyxy[0].cpu().numpy()

                                    if is_seg:
                                        poly = res.masks[j].xy[0].astype(np.int32)
                                        M = cv2.moments(poly)
                                        cx, cy = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"]!=0 else ((b[0]+b[2])/2,(b[1]+b[3])/2)
                                    else:
                                        cx, cy = (b[0]+b[2])/2,(b[1]+b[3])/2
                                        poly = None

                                    # ===== ARENA ASSIGNMENT =====
                                    for a_idx, arena in enumerate(all_arenas):

                                        if arena['type'] == 'rect':
                                            ax, ay, aw, ah = arena['x'], arena['y'], arena['w'], arena['h']
                                            inside = (ax <= cx <= ax + aw and ay <= cy <= ay + ah)

                                        elif arena['type'] == 'circle':
                                            cx0, cy0, r = arena['cx'], arena['cy'], arena['r']
                                            inside = ((cx - cx0)**2 + (cy - cy0)**2) <= r**2

                                        else:
                                            inside = False

                                        if inside:
                                            dets_by_arena[a_idx].append({
                                                'pos':(cx,cy),
                                                'conf':float(box.conf[0]),
                                                'box':b,
                                                'cid':int(box.cls[0]),
                                                'poly':poly
                                            })
                                            break

                                # ===== TRACKING + DRAWING =====
                                for a_idx, dets in enumerate(dets_by_arena):
                                    dets.sort(key=lambda x:x['conf'], reverse=True)
                                    dets = dets[:self.config['max_n']]

                                    if not tracks[a_idx]:
                                        tracks[a_idx] = {i:d['pos'] for i,d in enumerate(dets)}
                                        matches = [(i,i) for i in range(len(dets))]
                                    else:
                                        t_ids = list(tracks[a_idx].keys())
                                        if len(dets)>0:
                                            cost = np.array([
                                                [np.hypot(
                                                    tracks[a_idx][tid][0]-d['pos'][0],
                                                    tracks[a_idx][tid][1]-d['pos'][1]
                                                ) for d in dets] for tid in t_ids
                                            ])
                                            ri, ci = linear_sum_assignment(cost)
                                            matches = list(zip(ri,ci))
                                        else:
                                            matches = []

                                    for r,c in matches:
                                        tid, d = r, dets[c]
                                        tracks[a_idx][tid] = d['pos']

                                        cx, cy = d['pos']
                                        raw_data.append([
                                            frame_idx, a_idx+1, tid+1,
                                            class_names[d['cid']], cx, cy, d['cid']
                                        ])

                                        if writer:
                                            color = self.vibrant_palette[d['cid'] % len(self.vibrant_palette)]

                                            if is_seg and d['poly'] is not None:
                                                overlay = frame.copy()
                                                cv2.fillPoly(overlay, [d['poly']], color)
                                                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

                                            x1,y1,x2,y2 = d['box'].astype(int)
                                            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                                            cv2.circle(frame,(int(cx),int(cy)),2,(0,0,255),-1)

                                            label = f"A{a_idx+1}:ID{tid+1} | {class_names[d['cid']]}"
                                            (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                                            cv2.rectangle(frame,(x1,y1-th-10),(x1+tw+10,y1),color,-1)
                                            cv2.putText(frame,label,(x1+5,y1-5),
                                                        cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255),1)

                                if writer:
                                    writer.write(frame)

                                frame_idx += 1
                                frame_counter += 1

                            except Exception as e:
                                self._log_error_to_csv(base_n, "frame_processing", e)
                                continue

                        # ===== LOGGING =====
                        if time.time() - fps_timer > 1:
                            fps_val = frame_counter / (time.time() - fps_timer)
                            progress = int((frame_idx / reader.total_frames) * 100)
                            eta = ((time.time()-start_time)/frame_idx)*(reader.total_frames-frame_idx) if frame_idx else 0

                            self.log_signal.emit(
                                f"⚡ FPS: {fps_val:.2f} | ⏳ ETA: {eta:.1f}s | 📊 {progress}%"
                            )
                            self.progress_signal.emit(progress, base_n)

                            fps_timer = time.time()
                            frame_counter = 0

                    # ===== ANALYTICS =====
                    try:
                        if raw_data:
                            self.log_signal.emit("📊 Generating analytics...")
                            self._generate_analytics(base_n, raw_data, bg_frame, out_dir)
                    except Exception as e:
                        self._log_error_to_csv(base_n, "analytics", e)

                except Exception as e:
                    self._log_error_to_csv(base_n, "critical", e)

                finally:
                    if reader: reader.release()
                    if writer: writer.release()
                    gc.collect()

            self.finished_signal.emit("✅ Pipeline Completed Successfully")

        except Exception as e:
            self.finished_signal.emit(f"❌ Error: {str(e)}")


    # ==========================================
    # 📊 ANALYTICS (FULL RESTORED)
    # ==========================================
    def _generate_analytics(self, base_n, data, bg, out):
        df = pd.DataFrame(data, columns=["Frame", "Arena", "ID", "Class", "X", "Y", "CID"])
        df.to_csv(os.path.join(out, f"{base_n}_coordinates.csv"), index=False)

        # ===== TRAJECTORIES =====
        if self.config.get('save_traj') and bg is not None:
            canvas = bg.copy()
            for a_id in sorted(df['Arena'].unique()):
                for o_id in sorted(df[df['Arena']==a_id]['ID'].unique()):
                    subset = df[(df['Arena']==a_id)&(df['ID']==o_id)]
                    pts = subset[['X','Y']].values.astype(np.int32)

                    random.seed(int(a_id*100 + o_id))
                    color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))

                    if len(pts)>1:
                        cv2.polylines(canvas,[pts],False,color,1,cv2.LINE_AA)

            cv2.imwrite(os.path.join(out,f"{base_n}_trajectories.png"), canvas)

        # ===== HEATMAP =====
        if self.config.get('save_heat') and bg is not None:
            h,w = bg.shape[:2]
            accum = np.zeros((h,w),dtype=np.float32)

            for _,r in df.iterrows():
                cv2.circle(accum,(int(r['X']),int(r['Y'])),12,0.4,-1)

            accum = cv2.normalize(
                cv2.GaussianBlur(accum,(51,51),0),
                None,0,255,cv2.NORM_MINMAX
            ).astype(np.uint8)

            heatmap = cv2.applyColorMap(accum, cv2.COLORMAP_JET)

            cv2.imwrite(
                os.path.join(out,f"{base_n}_heatmap.png"),
                cv2.addWeighted(bg,0.6,heatmap,0.4,0)
            )

        # ===== EXCEL =====
        if self.config.get('save_xlsx'):
            xlsx_path = os.path.join(out, f"{base_n}_analysis.xlsx")

            with pd.ExcelWriter(xlsx_path) as writer:
                df.to_excel(writer, sheet_name="Master_Data", index=False)

                for a_id in sorted(df['Arena'].unique()):
                    arena_df = df[df['Arena']==a_id]
                    if not arena_df.empty:
                        arena_df.to_excel(
                            writer,
                            sheet_name=f"Arena_{int(a_id)}",
                            index=False
                        )