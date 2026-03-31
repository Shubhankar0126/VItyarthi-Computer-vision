# Smart Surveillance with YOLOv8 and DeepSORT

## Project Report

**Course:** Computer Vision  
**Project Title:** Smart Surveillance with YOLOv8 and DeepSORT  
**Student:** Mayank Pandey  
**Repository:** https://github.com/MAYANK479/Smart-Surveillance-Project

## Abstract

This project presents a real-time smart surveillance system built using modern computer vision techniques. The system detects people in video streams using YOLOv8, tracks them across frames using DeepSORT, estimates scene activity through frame-difference-based motion analysis, and logs crowd statistics for later review. The objective is to demonstrate how object detection and multi-object tracking can be integrated into a practical surveillance workflow that works on webcam input as well as stored video files.

The project emphasizes a simple but complete end-to-end pipeline rather than large-scale model training. It includes detection, tracking, alert generation, output video creation, logging, visualization of people counts over time, and an optional evaluation utility for labeled prediction files. The final result is a compact academic mini-project that shows a practical application of computer vision for intelligent monitoring.

## 1. Introduction

Video surveillance is widely used in public spaces, campuses, offices, transportation systems, and security-sensitive environments. Traditional CCTV systems rely heavily on human operators, which makes continuous monitoring difficult and inefficient. Computer vision can improve this process by automatically detecting and tracking people, estimating crowd density, and highlighting unusual situations for faster response.

This project focuses on a smart surveillance use case where the system identifies people present in a scene, keeps track of unique individuals using object tracking, and produces simple crowd analytics. The project combines real-time object detection with multi-object tracking so that the system can monitor dynamic scenes more effectively than frame-by-frame detection alone.

## 2. Problem Statement

Manual surveillance is time-consuming and does not scale well in environments with continuous video feeds. A useful computer vision system should be able to:

- detect people accurately in each frame,
- maintain identities across consecutive frames,
- estimate scene activity,
- raise a basic alert when the visible crowd exceeds a threshold, and
- store output in a form that can be reviewed later.

This project addresses that need by building an automated surveillance pipeline for person detection and tracking.

## 3. Objectives

The main objectives of the project are:

- to detect people in real time from webcam or video input,
- to track multiple people across frames using a tracking-by-detection approach,
- to calculate the current number of visible people and total unique tracked people,
- to estimate motion intensity in the scene,
- to generate an alert when crowd size crosses a configurable threshold,
- to save annotated video and structured logs for later analysis, and
- to provide a clean GitHub-ready academic submission with documentation and reproducible scripts.

## 4. Tools and Technologies Used

- **Programming Language:** Python
- **Computer Vision Library:** OpenCV
- **Detection Model:** YOLOv8 from Ultralytics
- **Tracking Algorithm:** DeepSORT
- **Data Handling:** Pandas
- **Visualization:** Matplotlib
- **Evaluation Support:** scikit-learn
- **Development Format:** Jupyter Notebook and Python scripts

## 5. Methodology

The project follows a tracking-by-detection pipeline:

1. Video frames are captured from a webcam or a video file.
2. Frames are resized for efficient processing.
3. YOLOv8 performs person detection at a configurable interval.
4. Detected bounding boxes are passed to DeepSORT for identity-aware tracking.
5. Confirmed tracks are drawn on the output frame with track IDs.
6. Frame-difference analysis is applied to estimate motion level in the scene.
7. Crowd statistics are calculated for each frame and written to a CSV-style log file.
8. An alert message is displayed when the current number of people exceeds the crowd threshold.
9. The annotated video is saved for inspection and the log can later be visualized as a trend plot.

![Project workflow](assets/workflow-diagram.png)

## 6. System Implementation

The implementation is organized into a few main files:

- `run_surveillance.py` runs the complete surveillance pipeline.
- `detection.ipynb` contains notebook-based experimentation and demonstration.
- `plot_people_log.py` reads the generated log file and plots people-count trends over time.
- `evaluate_predictions.py` computes accuracy, precision, recall, F1-score, and confusion matrix from a labeled CSV file.
- `README.md` and `PROJECT_ABSTRACT.md` provide project-level documentation.

### 6.1 Person Detection

YOLOv8 is used because it is fast, practical, and widely adopted for real-time object detection tasks. In this project, only detections belonging to the `person` class are retained. A confidence threshold is used to reduce weak detections.

### 6.2 Multi-Object Tracking

DeepSORT associates detections across frames and assigns track IDs to maintain identity continuity. This allows the system to estimate both:

- the number of people currently visible in the frame, and
- the total number of unique tracked people observed so far.

### 6.3 Motion Estimation

Motion estimation is implemented using frame differencing. Consecutive frames are converted to grayscale, blurred to reduce noise, subtracted to compute difference, thresholded, and processed with morphological closing. The resulting motion score gives a simple estimate of scene activity.

### 6.4 Logging and Output

For each processed frame, the system stores:

- frame number,
- current visible people,
- total unique tracked people, and
- motion level.

The pipeline also writes an annotated output video where detections, IDs, counts, motion level, FPS, and crowd alerts are displayed.

## 7. Results and Output

The system successfully produces the following outputs:

- an annotated surveillance video (`output.avi`),
- a structured activity log (`log.txt`),
- a people-count trend plot generated from the log,
- a notebook workflow for demonstration, and
- an optional evaluation module for labeled prediction data.

An example annotated frame from the project is shown below.

![Annotated output example](assets/demo-frame-final.png)

The project is primarily a functional system demonstration. In its current repository form, a labeled benchmark dataset for quantitative accuracy measurement is not bundled by default. Instead of reporting unsupported or artificial metrics, the repository includes an evaluation script and template CSV so that proper metrics can be computed once labeled predictions are prepared.

## 8. Strengths of the Project

- Uses a modern deep learning detector suitable for real-time applications.
- Combines detection and tracking in a practical end-to-end workflow.
- Supports both webcam input and stored video input.
- Generates visual as well as numeric outputs for analysis.
- Includes configurable detection interval and crowd threshold.
- Provides reusable scripts instead of only a notebook-based prototype.
- Maintains academic honesty by keeping evaluation optional until labeled data is available.

## 9. Limitations

Despite achieving the intended objective, the project has a few limitations:

- It currently focuses only on person detection, not broader activity understanding.
- Crowd alerts are threshold-based and do not model complex abnormal behavior.
- Performance depends on hardware capability and video quality.
- Long-term identity tracking may degrade in cases of severe occlusion or dense crowds.
- Quantitative evaluation depends on availability of labeled ground-truth files.

## 10. Future Scope

The project can be extended in several ways:

- add zone-based intrusion detection,
- detect suspicious or abnormal events,
- support more robust evaluation on annotated datasets,
- export results to dashboards for live monitoring,
- improve tracking in crowded scenes using stronger re-identification features, and
- deploy the system on edge devices for smart campus or smart city use cases.

## 11. Conclusion

This project demonstrates a practical smart surveillance pipeline using YOLOv8, DeepSORT, and OpenCV. It shows how real-time person detection, tracking, crowd counting, motion estimation, and alert generation can be combined into one compact application. The project meets the goals of a computer vision course submission by presenting a working end-to-end system, clear implementation structure, reusable scripts, and documented outputs.

Overall, the work highlights how computer vision can transform passive CCTV feeds into active, analytics-enabled monitoring systems. With further evaluation and deployment-oriented improvements, the project can be extended into a stronger real-world surveillance solution.

## 12. References

1. Ultralytics. YOLOv8 Documentation.  
2. Wojke, N., Bewley, A., and Paulus, D. Deep SORT: Deep Cosine Metric Learning for Real-Time Tracking.  
3. OpenCV Documentation.  
4. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python.  
5. McKinney, W. Data Structures for Statistical Computing in Python.
