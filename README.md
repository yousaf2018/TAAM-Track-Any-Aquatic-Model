# TAAM: Track Any Aquatic Model

**TAAM** is a high-performance, professional-grade desktop application
designed for the **automated tracking and behavioral analysis of aquatic
animals** (Zebrafish, Medaka, etc.) in laboratory environments. It bridges the gap between **massive AI foundation models (SAM 3)** and
**hyper-fast edge trackers (YOLO)** into a single, intuitive pipeline.

![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/AI-SAM3%20%2B%20YOLO-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
<p align="center">
  <img src="(https://github.com/yousaf2018/TAAM-Track-Any-Aquatic-Model/blob/main/TAAM_Workspace/assets/Logo.png)" alt="EthoGrid Logo" width="200">
</p>

## Acknowledgements

This application was developed in the **[Laboratory of Professor Chung-Der Hsiao](https://cdhsiao.weebly.com/pi-cv.html)** in collaboration with **Chung Yuan Christian University, Taiwan 🇹🇼**. Special credit and sincere gratitude are extended to **Professor Hsiao**, who shared his extensive research experience in biology and multiple domains, providing invaluable guidance and supervision throughout the development of this application.

<p align="center">
  <a href="https://www.cycu.edu.tw/">
    <img src="https://raw.githubusercontent.com/yousaf2018/EthoGrid/main/images/cycu.jpg" alt="Chung Yuan Christian University Logo" width="250">
  </a>
</p>

![TAAM Tool Overview](https://github.com/yousaf2018/TAAM-Track-Any-Aquatic-Model/blob/main/TAAM_Workspace/assets/TAAM-GUI.png)
*A snapshot of the TAAM interface*

# 🌟 Key Features

## ⚡ High-Speed Pipeline

Automatically trains a **custom YOLO model** from just a few
user-defined annotation clicks.

## 🧠 Teacher-Student AI Architecture

Uses **Segment Anything Model 3 (SAM 3)** as a **Teacher** to generate
large, accurate datasets, and **YOLO** as a **Student** for **high-speed
inference (100+ FPS)**.

## 📦 Batch Processing

Load, split, and annotate **dozens of videos simultaneously**.

## 🔬 Scientific Data Export

Generates high-precision CSV files containing:

-   Centroids (X, Y)
-   Pixel Area
-   Global Frame IDs
-   Full Mask Polygon Coordinates

## 🛠️ VRAM Optimization

Implements **automatic video chunking and CPU offloading** to handle
long **4K videos on standard 12GB GPUs** without **Out-of-Memory (OOM)**
crashes.

## 🖥️ Professional UI

Modern **dark-mode interface** with:

-   Side-by-side logging panels\
-   Project events monitoring\
-   AI engine terminal output\
-   Interactive hand-cursor navigation

------------------------------------------------------------------------

# 🚀 The 4-Step Workflow

TAAM is designed for **researchers with zero coding experience**.

## 1️⃣ PRE-PROCESS

Split massive laboratory recordings into manageable segments\
(e.g., **60-second chunks**) to optimize memory and processing speed.

## 2️⃣ ANNOTATE

Navigate to **3--5 frames** and draw a **bounding box** around the
target animals.\
TAAM records these as **few-shot prompts**.

## 3️⃣ TRAIN

Launch the automated pipeline.

SAM 3 will:

-   Propagate annotations across the whole video
-   Build a randomized training dataset
-   Train a **custom YOLO model (Detection or Segmentation)**

## 4️⃣ DEPLOY

Select your **versioned model** from the sidebar and run **high-speed
tracking** on any new video source.

------------------------------------------------------------------------

# 🏗️ Project Structure

For TAAM to work, the **SAM 3 repository must be placed in the root
directory**.

    TAAM_Workspace/
    │
    ├── main.py                    # Application Entry Point
    ├── sam3/                      # SAM 3 Repository (Root Directory)
    │
    ├── ui/
    │   └── video_selector.py      # High-performance drawing canvas
    │
    ├── backend/
    │   └── engine.py              # Core AI & Mathematical Logic
    │
    ├── workers/
    │   ├── pipeline_worker.py     # Threading for AI tasks
    │   └── splitter_worker.py     # Threading for pre-processing
    │
    ├── utils/
    │   └── stream_logger.py       # Terminal-to-GUI Bridge
    │
    └── TAAM_Workspace/            # User-defined Data Storage
                                   # (CSVs, Models, Videos)

------------------------------------------------------------------------

# 🛠️ Installation

## 1️⃣ Requirements

  Component   Requirement
  ----------- ----------------------------------------------------------
  OS          Windows 10/11 or Ubuntu 20.04+
  GPU         NVIDIA GPU (Minimum **8GB VRAM**, **12GB+ recommended**)
  Python      3.10 or 3.11

------------------------------------------------------------------------

## 2️⃣ Setup Environment

``` bash
# Clone the repository
git clone https://github.com/YourUsername/TAAM.git
cd TAAM

# Create virtual environment
conda create -n taam python=3.10
conda activate taam

# Install dependencies
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 3️⃣ Add SAM 3

Ensure your **local copy of the SAM 3 repository** is placed inside the
root folder:

    /sam3

------------------------------------------------------------------------

# 📊 Scientific Output

TAAM ensures the generated data is **ready for publication-level
analysis**.

## Precision CSV

  ------------------------------------------------------------------------------------------
  Global_Frame_ID   Object_ID   Centroid_X   Centroid_Y   Size_Pixels   Polygon_Coords
  ----------------- ----------- ------------ ------------ ------------- --------------------
  1240              1           845.2        412.5        12400         800,400;810,405...

  ------------------------------------------------------------------------------------------

## Annotated Video

A rendered **.mp4 video** featuring:

-   Semi-transparent polygon masks
-   ID tracking labels
-   Visual validation of AI performance

------------------------------------------------------------------------

# 🖥️ UI Screenshots

*(Add screenshots here)*

### Annotation View

High-precision **bounding box manager** with **Ctrl+C / Ctrl+V**
support.

### Engine Monitor

Real-time **TQDM progress bars** and **GPU status streamed directly to
the GUI**.

------------------------------------------------------------------------

# 🤝 Contributing

Contributions to improve:

-   Aquatic tracking logic
-   Additional pre-trained models
-   Performance optimization

are welcome.

Please **open an issue** or **submit a pull request**.

------------------------------------------------------------------------

# 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for details.

------------------------------------------------------------------------

# 👨‍🔬 Author

Developed by **\[Your Name / Lab Name\]**

**Empowering aquatic research with accessible, high-performance AI.**
