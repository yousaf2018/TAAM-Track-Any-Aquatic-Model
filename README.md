
# TAAM: Track Any Aquatic Model

**TAAM** is a high-performance, professional-grade desktop application designed for the automated tracking and behavioral analysis of aquatic animals (Zebrafish, Medaka, Daphnia, etc.) in laboratory environments.

TAAM bridges the gap between large AI foundation models and real-time edge AI, combining:

🧠 **SAM 3 (Teacher Model)** — few-shot learning & automatic dataset generation  
⚡ **YOLO (Student Model)** — ultra-fast inference (100+ FPS) for real-time tracking

![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/AI-SAM3%20%2B%20YOLO-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

<p align="center">
  <img src="https://github.com/yousaf2018/TAAM-Track-Any-Aquatic-Model/blob/main/source_code/assets/Logo.png" alt="TAAM Logo" width="200">
</p>

![TAAM GUI Snapshot](https://github.com/yousaf2018/TAAM-Track-Any-Aquatic-Model/blob/main/source_code/assets/TAAM-GUI.png)

---


## Acknowledgements

This application was developed in the **[Laboratory of Professor Chung-Der Hsiao](https://cdhsiao.weebly.com/pi-cv.html)** in collaboration with **Chung Yuan Christian University, Taiwan 🇹🇼**. Special credit and sincere gratitude are extended to **Professor Hsiao**, who shared his extensive research experience in biology and multiple domains, providing invaluable guidance and supervision throughout the development of this application.

<p align="center">
  <a href="https://www.cycu.edu.tw/">
    <img src="https://raw.githubusercontent.com/yousaf2018/EthoGrid/main/images/cycu.jpg" alt="Chung Yuan Christian University Logo" width="250">
  </a>
</p>

# 🌟 Key Features

- **Fully Automated AI Pipeline:** Train YOLO from just a few annotation clicks
- **Teacher–Student Architecture:** SAM 3 generates accurate datasets → YOLO learns fast tracking
- **YOLO-Only Mode:** Use existing SAM3 outputs to train YOLO directly
- **Batch Video Processing:** Process dozens or hundreds of videos automatically
- **Scientific Data Export:** High-precision CSVs with centroids, area, frame IDs, polygons, and image paths
- **VRAM Optimization:** Video chunking + CPU offload for stable 4K processing
- **Professional UI:** Dark-mode interface, side-by-side logging, project event monitoring

---

# 🚀 TAAM Workflow

## 1️⃣ PRE-PROCESSING
Split large lab videos into manageable segments for stable GPU processing.

## 2️⃣ ANNOTATION
Draw bounding boxes on a few frames. TAAM records them as few-shot prompts.

## 3️⃣ AI TRAINING PIPELINE

### 🧠 SAM 3 Stage
- Propagates masks across videos
- Tracks objects frame-by-frame
- Exports scientific CSV measurements
- Extracts frames to **sampling pool**: `Datasets/model_name/sampling_pool/`

> **Note:** Images are named to preserve video traceability: `OriginalVideoName_frame_000123.jpg`

### 🎯 YOLO Stage
- Cleans old train/val/test splits (but keeps `sampling_pool` intact)  
- Rebuilds dataset splits from sampling pool  
- Skips deleted SAM outputs  
- Handles first-time YOLO runs even if train/val/test folders do not exist  
- Supports Detection or Segmentation training automatically

## 4️⃣ DEPLOYMENT
Use trained models for high-speed tracking on new videos.

---

# 🗂️ Dataset Structure

```
Datasets/
└── model_name/
    ├── sampling_pool/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
```

---

# 🛠️ Installation

```bash
git clone https://github.com/YourUsername/TAAM.git
cd TAAM
conda create -n taam python=3.10
conda activate taam
pip install -r requirements.txt
```

---

# 📄 License
MIT License

---

# 👨‍🔬 Author
Mahmood Yousaf  
PhD Researcher — Biomedical Engineering  
Chung Yuan Christian University, Taiwan
