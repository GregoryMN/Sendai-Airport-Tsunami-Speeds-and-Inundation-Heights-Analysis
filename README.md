# Sendai-Airport-Tsunami-Speeds-and-Inundation-Heights-Analysis
Zip files and Python code associated with the analysis done on my Razer Blade 17 (2022). Laptop specifications include; Windows 11 Home, Intel(R) Core(TM) i7-12800H (2.40 GHz), 64GB RAM, and an NVIDIA GeForce RTX 3070 Ti Laptop GPU (8GB VRAM).

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)  
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

# 📋 Project Overview

This repository contains the complete, reproducible Python pipeline developed for the MSc dissertation “Automated Estimation of Tsunami Speed and Inundation Heights in the 2011 Sendai Tsunami Using Advanced Computer Vision”.

The code implements the methodology of McDonough-Margison et al. (2023) on the 2011 Tohoku Tsunami – Sendai Airport Terminal video, using:

YOLO11m-seg + BoT-SORT for vehicle detection/tracking
Custom TFLite orientation classifier + symmetry fallback
Kalman filter + Farneback optical-flow fusion for speed
Fine-tuned YOLO11 detection model for 5-class inundation-height mapping (0–2 m)

All scripts, graphs, CSVs and validation logs match exactly the versions reproduced in the dissertation appendices (D–E).
Scientific outputs (auto-generated in runs/detect/exp_*):

Annotated 1080p video with labels
6 publication-ready graphs (300 dpi) + CSV data tables
Validation logs for direct inclusion in Results/Appendix

---

## ✨ Features

### `speed_predict.py`
- YOLO11m-seg segmentation + BoT-SORT tracking
- Intelligent orientation detection (front/back/side) using local **TFLite** classifier (15-class model)
- Hybrid speed estimation: **Kalman filter** + **Farneback optical flow** fusion
- Dynamic metres-per-pixel (MPP) calculation adjusted for perspective & vehicle view
- Symmetry + aspect-ratio fallback for low-confidence crops

### `height_predict.py`
- Fine-tuned YOLO11 model (`best.pt`) for 5-class flood-level classification
- Direct mapping to physical inundation heights (0.0–2.0 m) based on literature categories
- Same visual styling and graph suite as speed script

**Both scripts** produce identical output structure for easy comparison within your Results.

---

tsunami-video-analysis/
├── fine_tune.py                          
├── height_predict.py                     
├── speed_predict.py                      
├── args.yaml                             
├── requirements.txt
├── README.md                             # This file
├── models/
│   ├── yolo11m-seg.pt
│   ├── best.pt                           # Fine-tuned (from training)
│   └── vehicle_orientation.tflite
├── input_videos/
│   └── 2011 Japan Tsunami - Sendai Airport Terminal. (Full Footage)_1080p.mp4
├── runs/detect/                          # Auto-generated outputs
│   ├── exp_height/
│   └── exp_speed/
└── docs/                                 

---

## 🛠️ Prerequisites

### Hardware
- NVIDIA GPU with ≥8 GB VRAM (strongly recommended – runs ~8× faster)
- CUDA 11.8 / cuDNN 8.6+ (or CPU fallback – much slower)

### Software
- Python 3.10–3.12 (tested on 3.11)
- FFmpeg (for video encoding – install via `conda` or system package manager)
- Git

---

## 🚀 Installation and Environment Setup

NVIDIA GPU (≥8 GB VRAM recommended; tested on RTX 3070 Ti)
CUDA 11.8+ / cuDNN 8.6+
Python 3.10–3.12
FFmpeg (for video encoding)
Git
~20 GB free disk space (models + video)

### 1. Clone the repository
```bash
git clone https://github.com/GregoryMN/Sendai-Airport-Tsunami-Speeds-and-Inundation-Heights-Analysis.git
cd tsunami-video-analysis
```

### 2. Create and Activate the Conda Environment
```
conda create -n tsunami-analysis python=3.11 -y
conda activate tsunami-analysis
```
### 3. Install Required Python Packages
```
pip install --upgrade pip
pip install -r requirements.txt
```

requirements.txt (copy-paste this into the file):
```
ultralytics==8.3.40
opencv-python==4.10.0.84
numpy==1.26.4
torch==2.4.1+cu118
torchvision==0.19.1+cu118
torchaudio==2.4.1+cu118
scikit-image==0.23.2
tensorflow==2.15.0
matplotlib==3.9.2
scipy==1.13.1
pandas==2.2.2
```

### 4. Set Environment Variables for CUDA
```
# Windows (PowerShell)
$env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
# Linux
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

### 5. Fine-Tuning Script (fine_tune.py)
Run once (≈4–6 hours on RTX 3070 Ti):
```
python fine_tune.py
```

Output model saved to runs/detect/fine_tuned_submerged_detection/weights/best.pt.


### 6. Download Models and Place Input Video
```
# YOLO11m-seg (auto-downloads on first run)
python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"

# Place these manually in /models/:
# 1. best.pt (your fine-tuned model from runs/detect/fine_tuned_submerged_detection/weights/)
# 2. vehicle_orientation.tflite
# 3. Input video in /input_videos/
```

### 7. Navigate to Project Directory and Run Scripts
```
cd tsunami-video-analysis
python height_predict.py
python speed_predict.py
```

### 8. Troubleshooting
CUDA out-of-memory: Reduce batch_size=16 or imgsz=480
TFLite error: Ensure tensorflow==2.15.0 (not 2.16+)
Video not found: Update hardcoded path in both scripts (or add argparse – see below)
No detections: Lower conf=0.3 in predict calls

## 📊 Outputs (Automatically Generated)
For each script:

runs/detect/exp_speed/output_video.mp4 (or exp_height)
6 high-resolution PNG graphs (300 dpi) perfect for dissertation figures
speed_validation_log.txt / validation_log.txt (ready for Appendix)

Graphs produced:

Histogram of speeds/heights
Average over time with error bars
Speed/height vs. Y-position (perspective validation)
Orientation / Flood-level pie chart
Vehicle-type / Level bar chart
Position × Speed/Height heatmap
