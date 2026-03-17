import os, cv2, torch, numpy as np, shutil, glob, csv, gc, sys, random, pandas as pd
from ultralytics import YOLO

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Locate SAM3
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM3_PATH = os.path.join(BASE_DIR, "sam3")
if SAM3_PATH not in sys.path: sys.path.append(SAM3_PATH)

# ==========================================
# GLOBAL SINGLETONS (Prevents VRAM Leaks)
# ==========================================
GLOBAL_SAM_MODEL = None
GLOBAL_SAM_PREDICTOR = None

class TAAMEngine:
    def __init__(self, workspace, config, log_app):
        self.workspace = os.path.abspath(workspace)
        self.config = config
        self.log_app = log_app
        self.predictor = None
        self.stop_flag = False 
        self.colors = [(20, 255, 57), (50, 50, 255), (255, 100, 50), (0, 255, 255), (255, 0, 255)]

    def load_sam3(self):
        """
        Loads SAM 3 into memory ONCE. 
        If it's already loaded, it reuses the global instance.
        """
        global GLOBAL_SAM_MODEL, GLOBAL_SAM_PREDICTOR
        
        self.log_app.emit("AI: Validating GPU Model Context...")
        if GLOBAL_SAM_MODEL is not None and GLOBAL_SAM_PREDICTOR is not None:
            self.log_app.emit("AI: ♻️ Model already exists in VRAM. Reusing singleton.")
            self.predictor = GLOBAL_SAM_PREDICTOR
            return True
            
        try:
            from sam3.model_builder import build_sam3_video_model
            self.log_app.emit("AI: 🧠 First Run: Initializing SAM 3 Architecture to CUDA...")
            with torch.inference_mode():
                GLOBAL_SAM_MODEL = build_sam3_video_model()
                GLOBAL_SAM_PREDICTOR = GLOBAL_SAM_MODEL.tracker
                GLOBAL_SAM_PREDICTOR.backbone = GLOBAL_SAM_MODEL.detector.backbone
            
            self.predictor = GLOBAL_SAM_PREDICTOR
            self.log_app.emit("AI: ✅ SAM 3 Model Loaded Successfully.")
            return True
        except Exception as e:
            self.log_app.emit(f"AI: ❌ SAM3 Load Failed: {e}")
            return False

    def clear_vram(self):
        """
        Purges temporary buffers but KEEPS the global model loaded.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def run_full_pipeline(self, video_paths, annotations_map, model_name, progress_callback):
        try:
            exp_root = os.path.join(self.workspace, "Experiments", model_name)
            ds_dir = os.path.abspath(os.path.join(self.workspace, "Datasets", model_name))
            pool_dir = os.path.join(ds_dir, "sampling_pool")
            os.makedirs(exp_root, exist_ok=True); os.makedirs(ds_dir, exist_ok=True); os.makedirs(pool_dir, exist_ok=True)

            video_info = {}
            for v_idx, v_path in enumerate(video_paths):
                if self.stop_flag: break
                v_name = os.path.splitext(os.path.basename(v_path))[0]
                self.log_app.emit(f"PROJECT: Tracking {v_name} ({v_idx+1}/{len(video_paths)})")
                v_exp_dir = os.path.join(exp_root, v_name)
                os.makedirs(v_exp_dir, exist_ok=True)
                csv_p = self._process_video_sam3(v_path, annotations_map.get(v_path, {}), v_exp_dir, pool_dir, progress_callback)
                if csv_p: video_info[v_path] = csv_p

            if self.stop_flag or not video_info: return False

            # Generate YOLO Dataset
            yaml_path = self._generate_yolo_dataset(video_info, ds_dir, progress_callback)
            
            if yaml_path and not self.stop_flag:
                # Clean VRAM (but keep SAM 3 Singleton active) before YOLO starts
                self.clear_vram()
                
                task = "segment" if self.config['task_type'] == "Segmentation" else "detect"
                cust_w = self.config.get('custom_weights')
                model_base = cust_w if (cust_w and os.path.exists(cust_w)) else ("yolov8n-seg.pt" if task == "segment" else "yolov8n.pt")
                
                self.log_app.emit(f"PROJECT: Initiating YOLOv8 Training (Base: {os.path.basename(model_base)})...")
                model = YOLO(model_base)
                model.train(data=yaml_path, epochs=self.config['epochs'], imgsz=self.config['imgsz'], batch=self.config['yolo_batch'],
                            project=os.path.join(self.workspace, "Models"), name=model_name, workers=0, device=0, task=task, exist_ok=True)
            
            # Clean up the massive image pool to save hard drive space
            shutil.rmtree(pool_dir, ignore_errors=True)
            return True
        except Exception as e:
            self.log_app.emit(f"PIPELINE EXCEPTION: {str(e)}")
            return False

    def _process_video_sam3(self, v_path, ann, out_dir, pool_dir, cb):
        temp, proc = os.path.join(out_dir, "temp"), os.path.join(out_dir, "proc")
        os.makedirs(temp, exist_ok=True); os.makedirs(proc, exist_ok=True)
        name = os.path.splitext(os.path.basename(v_path))[0]
        csv_p = os.path.join(out_dir, f"{name}_data.csv")
        vid_p = os.path.join(out_dir, f"{name}_SAM3_Annotated.mp4")
        
        with open(csv_p, 'w', newline='') as f: 
            csv.writer(f).writerow(["Global_Frame_ID", "Object_ID", "Class_ID", "Centroid_X", "Centroid_Y", "Size_Pixels", "Size_um2", "Polygon_Coords", "Image_Path"])

        cap = cv2.VideoCapture(v_path); fps, w, h = cap.get(5), int(cap.get(3)), int(cap.get(4))
        chunk_paths = self._split_video(cap, fps, w, h, temp); cap.release()

        # GUI Rects to Normalized Prompts
        prompts = {i+1: {'pt': (rect.center().x() / w, rect.center().y() / h), 'cls': cid} for i, (rect, cid) in enumerate(ann.get(0, []))}

        total_c = len(chunk_paths)
        for i, cp in enumerate(chunk_paths):
            if self.stop_flag or not prompts: break
            cb(10 + int((i/total_c)*40), f"SAM3 Batch {i+1}")
            save_p = os.path.join(proc, f"p_{os.path.basename(cp)}")
            prompts = self._track_chunk(cp, i, prompts, fps, csv_p, pool_dir, w, h, save_p)

        self._stitch(proc, vid_p, fps, w, h)
        shutil.rmtree(temp, ignore_errors=True); shutil.rmtree(proc, ignore_errors=True)
        return csv_p

    def _track_chunk(self, cp, c_idx, prompts, fps, csv_path, pool_dir, w_orig, h_orig, save_p):
        state = self.predictor.init_state(video_path=cp, offload_video_to_cpu=True)
        with torch.inference_mode():
            for oid, data in prompts.items():
                pt = data['pt'] 
                self.predictor.add_new_points(state, 0, oid, torch.tensor([[pt[0], pt[1]]], dtype=torch.float32, device="cuda"), torch.tensor([1], dtype=torch.int32, device="cuda"))
        
        last_masks, next_p, vis_data = {}, {}, {}
        cap_read = cv2.VideoCapture(cp); frame_count = int(cap_read.get(7))
        with open(csv_path, 'a', newline='') as f_csv:
            csv_w = csv.writer(f_csv)
            gen = self.predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=frame_count, reverse=False, propagate_preflight=True)
            for f_idx, oids, _, masks, _ in gen:
                if self.stop_flag: break
                ret, frame = cap_read.read()
                if not ret: break
                img_p = os.path.abspath(os.path.join(pool_dir, f"v_{c_idx}_{f_idx:04d}_{random.randint(100,999)}.jpg"))
                cv2.imwrite(img_p, frame); vis_data[f_idx] = {}
                for j, oid in enumerate(oids):
                    m = (masks[j] > 0.0).cpu().numpy().squeeze()
                    c, area = self._get_centroid_area(m)
                    if c:
                        poly = self._get_polygon_str(m)
                        csv_w.writerow([(c_idx*frame_count)+f_idx, oid, prompts[oid]['cls'], c[0], c[1], area, 0, poly, img_p])
                        vis_data[f_idx][oid] = {'pt': c, 'cls': prompts[oid]['cls']}
                        last_masks[oid] = m
                if f_idx % 20 == 0: self.log_app.emit(f"  > Computed frame {f_idx}...")

        # Reset SAM 3 state for the next chunk
        self.predictor.clear_all_points_in_video(state)
        cap_read.release()
        self.clear_vram()

        self._render(cp, save_p, fps, vis_data)
        
        for oid, m in last_masks.items():
            c, _ = self._get_centroid_area(m)
            if c: next_p[oid] = {'pt': (c[0]/w_orig, c[1]/h_orig), 'cls': prompts[oid]['cls']}
        return next_p

    def _render(self, inp, out, fps, data):
        cap = cv2.VideoCapture(inp); w, h = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        curr_f = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if curr_f in data:
                for oid, info in data[curr_f].items():
                    color = self.colors[oid % len(self.colors)]
                    cv2.circle(frame, info['pt'], 5, color, -1)
                    cv2.putText(frame, f"ID:{oid}", (info['pt'][0]+8, info['pt'][1]-8), 0, 0.6, (255,255,255), 2)
            writer.write(frame); curr_f += 1
        cap.release(); writer.release()

    def _get_centroid_area(self, mask):
        a = np.count_nonzero(mask)
        rows, cols = np.where(mask)
        return ((int(np.mean(cols)), int(np.mean(rows))), a) if len(rows) > 0 else (None, 0)

    def _get_polygon_str(self, mask):
        m8 = (mask.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return ""
        c = max(cnts, key=cv2.contourArea)
        return ";".join([f"{p[0][0]},{p[0][1]}" for p in cv2.approxPolyDP(c, 0.005*cv2.arcLength(c, True), True)])

    def _generate_yolo_dataset(self, video_info_map, dataset_root, cb):
        all_p = []
        for v_path, csv_p in video_info_map.items():
            df = pd.read_csv(csv_p); df.columns = df.columns.str.strip()
            for fid in df["Global_Frame_ID"].unique(): all_p.append((v_path, fid, csv_p))
        random.shuffle(all_p); sampled = all_p[:self.config['max_frames']]
        split_map = {p: ("train" if i < len(sampled)*(self.config['tr']/100) else ("val" if i < len(sampled)*((self.config['tr']+self.config['va'])/100) else "test")) for i, p in enumerate(sampled)}
        for s in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dataset_root, "images", s), exist_ok=True); os.makedirs(os.path.join(dataset_root, "labels", s), exist_ok=True)
        for i, (v_path, f_idx, csv_p) in enumerate(sampled):
            if self.stop_flag: return None
            split = split_map[(v_path, f_idx, csv_p)]
            df = pd.read_csv(csv_p); rows = df[df["Global_Frame_ID"]==f_idx]
            img_src = rows.iloc[0]["Image_Path"]
            shutil.copy(img_src, os.path.join(dataset_root, "images", split, os.path.basename(img_src)))
            with open(os.path.join(dataset_root, "labels", split, os.path.basename(img_src).replace(".jpg", ".txt")), "w") as f_lbl:
                cap = cv2.VideoCapture(v_path); fw, fh = int(cap.get(3)), int(cap.get(4)); cap.release()
                for _, r in rows.iterrows():
                    poly = str(r["Polygon_Coords"])
                    if self.config['task_type'] == "Segmentation":
                        f_lbl.write(f"0 " + " ".join([f"{float(p.split(',')[0])/fw:.6f} {float(p.split(',')[1])/fh:.6f}" for p in poly.split(';') if ',' in p]) + "\n")
                    else:
                        coords = [list(map(float, p.split(','))) for p in poly.split(';') if ',' in p]
                        if coords:
                            x, y = [c[0] for c in coords], [c[1] for c in coords]
                            bw, bh = max(x)-min(x), max(y)-min(y)
                            f_lbl.write(f"0 {(min(x)+bw/2)/fw:.6f} {(min(y)+bh/2)/fh:.6f} {bw/fw:.6f} {bh/fh:.6f}\n")
            cb(50 + int((i/len(sampled))*40), f"YOLO Dataset: {split}")
            
        yaml_p = os.path.join(dataset_root, "data.yaml")
        yaml_content = f"path: {dataset_root}\ntrain: {os.path.join(dataset_root, 'images', 'train')}\nval: {os.path.join(dataset_root, 'images', 'val')}\ntest: {os.path.join(dataset_root, 'images', 'test')}\nnames:\n  0: aquatic_animal"
        with open(yaml_p, "w") as f: f.write(yaml_content)
        return yaml_p

    def _split_video(self, cap, fps, w, h, out_dir):
        per = int(fps * self.config['chunk_duration']); paths, idx = [], 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            out_f = os.path.abspath(os.path.join(out_dir, f"c_{idx:03d}.mp4"))
            writer = cv2.VideoWriter(out_f, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            cnt = 0; writer.write(frame)
            while cnt < per - 1:
                ret, frame = cap.read(); 
                if not ret: break
                writer.write(frame); cnt += 1
            writer.release(); paths.append(out_f); idx += 1
        return paths

    def _stitch(self, d, out, fps, w, h):
        files = sorted(glob.glob(os.path.join(d, "*.mp4")))
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in files:
            cap = cv2.VideoCapture(f)
            while True:
                ret, frame = cap.read()
                if not ret: break
                writer.write(frame)
            cap.release()
        writer.release()