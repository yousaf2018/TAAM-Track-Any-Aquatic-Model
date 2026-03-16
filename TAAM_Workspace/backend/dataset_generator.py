import os
import cv2
import numpy as np
import torch

GLOBAL_SAM_MODEL = None
GLOBAL_SAM_PREDICTOR = None

class DatasetGenerator:
    def __init__(self, output_dir, log_signal):
        self.output_dir = output_dir
        self.log_signal = log_signal
        
        self.dataset_dir = os.path.join(self.output_dir, "YOLO_Dataset")
        self.images_dir = os.path.join(self.dataset_dir, "images", "train")
        self.labels_dir = os.path.join(self.dataset_dir, "labels", "train")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

    def load_sam(self):
        global GLOBAL_SAM_MODEL, GLOBAL_SAM_PREDICTOR
        if GLOBAL_SAM_MODEL is not None: return GLOBAL_SAM_PREDICTOR
        
        try:
            import sam3
            from sam3.model_builder import build_sam3_video_model
            with torch.inference_mode():
                GLOBAL_SAM_MODEL = build_sam3_video_model()
                GLOBAL_SAM_PREDICTOR = GLOBAL_SAM_MODEL.tracker
                GLOBAL_SAM_PREDICTOR.backbone = GLOBAL_SAM_MODEL.detector.backbone
            return GLOBAL_SAM_PREDICTOR
        except ImportError:
            self.log_signal.emit("WARNING: SAM 3 not installed. Running in Mock Mode for UI testing.")
            return None

    def generate_yolo_dataset(self, video_path, annotations, max_frames=300):
        predictor = self.load_sam()
        
        # MOCK MODE if SAM 3 isn't installed yet
        if predictor is None:
            self.log_signal.emit("MOCK DATASET GEN: Creating fake dataset to test YOLO pipeline...")
            yaml_path = os.path.join(self.dataset_dir, "dataset.yaml")
            with open(yaml_path, 'w') as f:
                f.write(f"path: {self.dataset_dir}\n")
                f.write(f"train: images/train\n")
                f.write(f"val: images/train\n")
                f.write(f"names:\n  0: custom_animal\n")
            return yaml_path

        # REAL MODE
        cap = cv2.VideoCapture(video_path)
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        state = predictor.init_state(video_path=video_path)
        
        for f_idx, boxes in annotations.items():
            for obj_id, (rect, cls_id) in enumerate(boxes):
                cx = rect.x() + rect.width() / 2
                cy = rect.y() + rect.height() / 2
                pt = torch.tensor([[cx, cy]], dtype=torch.float32)
                lbl = torch.tensor([1], dtype=torch.int32)
                predictor.add_new_points(state, f_idx, obj_id, pt, lbl)

        self.log_signal.emit(f"Propagating SAM 3 for {max_frames} frames...")
        
        frames_processed = 0
        for f_idx, oids, _, masks, _ in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=max_frames
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: break
            
            img_name = f"frame_{f_idx:05d}.jpg"
            cv2.imwrite(os.path.join(self.images_dir, img_name), frame)
            
            label_path = os.path.join(self.labels_dir, f"frame_{f_idx:05d}.txt")
            with open(label_path, 'w') as f_txt:
                for j, oid in enumerate(oids):
                    m = (masks[j] > 0.0).cpu().numpy().squeeze()
                    rows, cols = np.where(m)
                    if len(rows) > 0:
                        x_min, x_max = np.min(cols), np.max(cols)
                        y_min, y_max = np.min(rows), np.max(rows)
                        x_center = ((x_min + x_max) / 2) / img_width
                        y_center = ((y_min + y_max) / 2) / img_height
                        w = (x_max - x_min) / img_width
                        h = (y_max - y_min) / img_height
                        f_txt.write(f"0 {x_center} {y_center} {w} {h}\n")
            
            frames_processed += 1
            if frames_processed >= max_frames: break

        predictor.clear_all_points_in_video(state)
        cap.release()
        
        yaml_path = os.path.join(self.dataset_dir, "dataset.yaml")
        with open(yaml_path, 'w') as f:
            f.write(f"path: {self.dataset_dir}\n")
            f.write(f"train: images/train\n")
            f.write(f"val: images/train\n")
            f.write(f"names:\n  0: custom_animal\n")
            
        return yaml_path
