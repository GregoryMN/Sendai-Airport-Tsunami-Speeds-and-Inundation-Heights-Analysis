# Sendai-Airport-Tsunami-Speeds-and-Inundation-Heights-Analysis
Zip files and Python code associated with the analysis done on my Razer Blade 17 (2022). Laptop specifications include; Windows 11 Home, Intel(R) Core(TM) i7-12800H (2.40 GHz), 64GB RAM, and an NVIDIA GeForce RTX 3070 Ti Laptop GPU (8GB VRAM).

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)  
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

# 📋 Project Overview

This repository contains production-ready Python scripts that implement a novel computer-vision pipeline for extracting **tsunami flow velocity** (using vehicles as a proxy) and **inundation height** from civilian video footage.  

The code directly supports the methodology proposed in **McDonough-Margison et al. (2023)** and applies it to the **2011 Tohoku Tsunami – Sendai Airport Terminal** video (one of the clearest publicly available high-resolution recordings).  

**Key scientific outputs** (automatically generated):
- Annotated output video with speed/orientation/height labels
- Publication-quality graphs (histogram, time-series, scatter, pie, bar, heatmap)
- Validation logs for reproducibility

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
├── speed_predict.py
├── height_predict.py
├── requirements.txt
├── README.md
├── LICENSE
├── models/
│   ├── yolo11m-seg.pt                  # Auto-downloaded
│   ├── best.pt                         # Your fine-tuned flood model
│   └── vehicle_orientation.tflite      # Custom 15-class orientation model
├── input_videos/                       # Place your .mp4 here
├── runs/detect/
│   ├── exp_speed/                      # Auto-generated
│   └── exp_height/
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

## 🚀 Installation (5 minutes)

### 1. Clone the repository
```bash
git clone https://github.com/GregoryMN/Sendai-Airport-Tsunami-Speeds-and-Inundation-Heights-Analysis.git
cd tsunami-video-analysis
```

### 2. Create and activate virtual environment
```
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```
### 3. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt

requirements.txt (copy-paste this into the file):

ultralytics>=8.3.0
opencv-python>=4.10.0
numpy>=1.26.0
torch>=2.4.0
torchvision>=0.19.0
torchaudio>=2.4.0
scikit-image>=0.23.0
tensorflow>=2.15.0
matplotlib>=3.9.0
scipy>=1.13.0
```
### 4. Download models
```
# YOLO11m segmentation (auto-downloads on first run, or manual)
python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"

# Place these two files yourself:
#   1. Your fine-tuned flood model → models/best.pt
#   2. Orientation TFLite model → models/vehicle_orientation.tflite
```
### 5. Place input video
```
mkdir -p input_videos
# Download or copy: 2011 Japan Tsunami - Sendai Airport Terminal (Full Footage)_1080p.mp4
```
### 6. Run
```
python speed_predict.py
python height_predict.py
```
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
