
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

# TAAM Installation Guide

## 📋 Prerequisites

**Python:** Version 3.10 or 3.11 is highly recommended. (Avoid 3.12 for now due to library compatibility).

**GPU (Optional):** NVIDIA GPU with latest drivers for AI acceleration.

**Hugging Face Account:** You need a User Access Token (Read/Write) to download models.

---

## 🪟 Windows Installation (Recommended)

### Method 1: Automatic (The "Two-Click" Way)

This is the easiest method and includes built-in fixes for the "PyQt6 DLL Conflict" error.

**Step 1: Run INSTALLER.bat**  
Double-click this file. It will create the virtual environment, install PyTorch (CUDA), and prompt you for your Hugging Face token.

**Step 2: Run RUNNER.bat**  
Once installation is finished, always use this file to start the app. It isolates the environment to prevent errors caused by other software (like Anaconda).

---

### Method 2: Manual PowerShell (If BAT files fail)

If the `.bat` files do not open, use the manual terminal approach:

#### Open PowerShell in the project folder

Allow script execution (run this once):
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

#### Create and activate environment:
```powershell
python -m venv sam3_tracker_venv
.\sam3_tracker_venv\Scripts\Activate.ps1
```

#### Install dependencies:
```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-win.txt
python -m pip install -e ./sam3
```

#### Critical PyQt6 Fix (If you get DLL errors):
```powershell
python -m pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
python -m pip install PyQt6==6.6.1 PyQt6-Qt6==6.6.1 --no-cache-dir
```

---

## 🐧 Linux Installation



### Method 1: Manual Terminal

```bash
# 1. Create venv
python3 -m venv sam3_tracker_venv
source sam3_tracker_venv/bin/activate

# 2. Install Torch
pip install torch torchvision torchaudio

# 3. Install Requirements
pip install -r requirements.txt
pip install -e ./sam3

# 4. Login to Hugging Face
pip install huggingface_hub
huggingface-cli login
```

---

## 🛠 Troubleshooting (Windows)

### 1. Error: ImportError: DLL load failed while importing QtWidgets

This is the most common Windows error. It happens because other software (Anaconda, OBS, Drivers) is conflicting with Qt.

**Fix A:** Always use RUNNER.bat to launch the app. It hides conflicting system paths.

**Fix B:** Install the Microsoft Visual C++ Redistributable and restart your computer.

**Fix C:** Open RUNNER.bat and ensure the PATH isolation line is active.

---

### 2. Error: huggingface_hub.login not found

Fix: Your version of the hub is too old. Run:
```powershell
.\sam3_tracker_venv\Scripts\python.exe -m pip install --upgrade "huggingface_hub>=0.23.0"
```

---

### 3. Python is not recognized

Fix: During Python installation, you must check:

✔ "Add Python to PATH"

If missed:
- Uninstall Python
- Reinstall and enable PATH option

---

## 🚀 How to Run (Daily Use)

### Windows
Double-click **RUNNER.bat**. This is programmed to handle all path conflicts and launch `main.py` instantly.

### Linux
```bash
source sam3_tracker_venv/bin/activate
python main.py
```

---

## ✅ Verifying GPU Support

Once the app is running, verify GPU usage:

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
```



# 📄 License
MIT License

---

# 👨‍🔬 Author
Mahmood Yousaf  
PhD Researcher — Biomedical Engineering  
Chung Yuan Christian University, Taiwan
