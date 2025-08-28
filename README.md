💤 Anti-Sleep Alarm Detection
📌 Overview

This project is a real-time anti-sleep alarm system that uses computer vision techniques to detect signs of drowsiness (e.g., eye closure, yawning) and triggers an alert to prevent accidents caused by fatigue. Ideal for drivers or anyone who needs to stay alert during critical tasks.
🔧 Features

    👁️ Eye closure detection using [Dlib / Mediapipe / OpenCV]
    😮 Yawn detection via facial landmark analysis
    🔊 Alarm activation when drowsiness is detected
    📷 Real-time video processing via webcam
    📈 Adjustable sensitivity thresholds
    💻 Lightweight and works offline

🚀 How It Works

    Captures live webcam feed.
    Detects facial landmarks to monitor eyes and mouth.
    Computes Eye Aspect Ratio (EAR) and/or Mouth Aspect Ratio (MAR).
    Triggers an audible alarm when thresholds are breached for a sustained time.

🖥️ Tech Stack

    Python
    OpenCV
    Dlib or Mediapipe
    NumPy
    Pygame (for playing alarm sound)
