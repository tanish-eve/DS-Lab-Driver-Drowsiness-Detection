# Drowsiness Detection System



## Overview

The **Drowsiness Detection System** is a project designed to monitor a person's alertness in real-time by analyzing facial features.  
By utilizing computer vision and machine learning techniques, the system aims to detect signs of drowsiness and provide timely alerts — particularly useful for applications like driver monitoring.

This repository focuses on illustrating the full development process, including data capture, auto-labeling, model training, and detection pipeline integration.

---

## Features

- **Real-time Monitoring**: Detects signs of drowsiness using a webcam or video input.
- **Dual Model Detection**: Separate YOLOv8 models for eye closure detection and yawning detection.
- **Facial Landmarks Analysis**: Tracks eye status, head position, and mouth movements.
- **Data Capture Pipeline**: Tools to collect and organize custom datasets.
- **Auto Labeling with GroundingDINO**: Automated bounding box generation.
- **User Interface**: Built with PyQt5 for real-time visualization and alerts.

---

## Key Files

- `AutoLabelling.py`: Script for automated bounding box labeling using GroundingDINO.
- `CaptureData.py`: Records and logs video data for analysis or training.
- `DrowsinessDetector.py`: Core detection script integrating real-time inference and alerts.
- `LoadData.ipynb`: Loads and preprocesses datasets.
- `RedirectData.ipynb`: Organizes and redirects captured data for training.
- `train.ipynb`: Notebook for training the YOLO models.

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/tanish-eve/Real_time_drowsy_driving_detection.git
    cd Real_time_drowsy_driving_detection
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate 
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the detection system:**
    ```bash
    python DrowsinessDetector.py
    ```

---

## Usage

- **Real-Time Detection:** Run `DrowsinessDetector.py` with a connected webcam to monitor drowsiness.
- **Data Capture:** Use `CaptureData.py` to collect video frames for training or testing.
- **Training New Models:** Use `train.ipynb` to retrain the models on your custom datasets.

---

## How It Works

The system uses two separate YOLOv8 models:

1. **Eye Detection Model:**
   - Classifies eyes as open or closed.
   - Trained on public datasets:
     - [Eyes Dataset](https://www.kaggle.com/datasets/charunisa/eyes-dataset/code)
     - [MRL Eye Dataset](https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset)
   - ~53,000 images for training, ~3,000 images for validation.

2. **Yawning Detection Model:**
   - Detects yawning (mouth open) vs not yawning (mouth closed).
   - Trained on:
     - [Yawning Dataset](https://www.kaggle.com/datasets/deepankarvarma/yawning-dataset-classification?select=yawn)

**Auto Labeling:**  
GroundingDINO was used to generate bounding boxes for YOLO training to improve dataset quality.

Once trained, the models' predictions are combined with confidence thresholds and visualized in a PyQt5 GUI.

---

## Technologies Used

- **Python**
- **YOLOv8** – Object detection framework.
- **OpenCV** – Computer vision tasks.
- **GroundingDINO** – Auto-labeling tool.
- **TensorFlow / Keras** – Model training.
- **PyQt5** – Graphical user interface.

---

## Important Note

This repository is intended primarily to showcase the development process of a drowsiness detection system — including data collection, model training, and real-time integration.

The uploaded model weights are **preliminary** and **not fully trained to convergence**.  
They are mainly for demonstration purposes, and final production-ready models are maintained separately.

We appreciate any feedback and contributions to improve the system.

---

## Future Improvements

- **Integration with Wearables:** Add heart rate or other vitals monitoring.
- **Multi-Person Detection:** Extend detection to multiple subjects simultaneously.
- **Mobile Deployment:** Create a mobile app version for real-time on-the-go monitoring.

---