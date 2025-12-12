
# AI Fitness Trainer

An AI-powered computer vision application that acts as a personal gym trainer. It analyzes your exercise form in real-time using **MediaPipe Pose Estimation** and provides instant feedback on posture corrections and repetition counting.

## How to Run (Quick Start)

Follow these steps to set up and run the project on your machine.

### 1. Install Dependencies
Make sure you have Python installed.

You must create a new environment (to avoid dependency conflicts) and install the required libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 2. Run the Application
To start the analysis on the demo video, simply run:

```bash
python main.py
```

---

## ⚙️ Configuration (Switch Exercises)
You can switch between exercises or change the video input by editing the top section of `main.py`:

```python
# Open main.py and look for lines 8-12:

# 1. Select Exercise: Change to 'bicep' or 'lateral'
EXERCISE_TYPE = 'bicep'  

# 2. Select Video: Replace with your filename or use 0 for Webcam
VIDEO_PATH = 'demo_curl_.mp4' 
```

---

## Features & Exercises Supported

### Bicep Curl
*   **Rep Counter:** Counts reps based on full range of motion (Down > 150°, Up < 60°).
*   **Elbow Drift Check:** Warns "FIX ELBOW" if elbows move forward (Cheating).
*   **Back Stability:** Warns "FIX BACK" if the user swings their body.

### Lateral Raise
*   **Height Limit:** Warns "LOWER ARMS" if wrists go above shoulder level (prevents injury).
*   **Correct Hold:** Detects "Good Hold" when arms are at the correct angle.

## Tech Stack
*   **Python**
*   **MediaPipe** (Pose Estimation)
*   **OpenCV** (Image Processing)
*   **NumPy** (Geometric Calculations)

---
