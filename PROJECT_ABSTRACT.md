# Project Abstract

This project presents a smart surveillance pipeline built with computer vision techniques for real-time people detection and tracking. The system uses YOLOv8 for person detection, DeepSORT for multi-object tracking, and frame-difference motion analysis to estimate scene activity. It also logs crowd-related statistics such as current visible people and total unique tracked people over time.

The main goal of the project is to demonstrate how modern object detection and tracking can be combined into a simple surveillance workflow. The current implementation supports webcam or video-file input, creates annotated output video, generates CSV logs for later analysis, and provides optional evaluation support through labeled prediction files.

This repository is designed as an academic mini-project submission and focuses on clarity, reproducibility, and GitHub presentation. It highlights practical computer vision workflow design more than large-scale model training.
