# Smart Surveillance with YOLOv8 and DeepSORT

![Smart Surveillance Banner](assets/repo-banner-final.png)

This project is a computer vision mini-project for real-time smart surveillance. It detects people with YOLOv8, tracks them with DeepSORT, estimates motion between frames, logs crowd statistics, and raises a simple crowd alert.

The repo is now organized so it can be shown on GitHub more cleanly:

- `detection.ipynb`: notebook walkthrough and quick experimentation
- `run_surveillance.py`: reusable script for real-time detection and tracking
- `plot_people_log.py`: plots people-count trends from the generated CSV log
- `evaluate_predictions.py`: optional evaluation script for labeled prediction CSVs
- `evaluation_labels_template.csv`: template for adding `y_true` and `y_pred`
- `requirements.txt`: Python dependencies
- `archive/`: local dataset folder used only during experimentation
- `assets/`: README visuals such as banner, demo frame, and workflow diagram

## Academic Summary

A short academic-style abstract is available in `PROJECT_ABSTRACT.md`.

## Project Showcase

### Demo Frame

![Annotated Demo Frame](assets/demo-frame-final.png)

### Workflow Diagram

![Project Workflow](assets/workflow-diagram.png)

## Workflow

1. Run the surveillance pipeline on webcam or video input.
2. Save annotated output video and CSV logs.
3. Plot people-count trends from the log.
4. Optionally evaluate predictions with a labeled CSV file.

## Features

- Person detection using YOLOv8
- Multi-object tracking using DeepSORT
- Current visible people count
- Total unique tracked people count
- Basic frame-difference motion estimation
- Crowd alert based on configurable threshold
- CSV logging for later analysis

## Setup

Create and activate a Python 3.10 environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Download

This project was tested with the Kaggle dataset below:

`suryaprabhakaran2005/road-accidents-from-cctv-footages-dataset`

You can download it with `kagglehub` using:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "suryaprabhakaran2005/road-accidents-from-cctv-footages-dataset"
)

print("Path to dataset files:", path)
```

The printed `path` is the local dataset folder on your machine, and it may differ from system to system.

## Run the Project

### Option 1: Notebook

```bash
jupyter notebook detection.ipynb
```

### Option 2: Python script

Use webcam:

```bash
python run_surveillance.py --source 0
```

Use a video file:

```bash
python run_surveillance.py --source path/to/video.mp4 --no-display
```

This creates:

- `output.avi`: annotated output video
- `log.txt`: CSV log with frame, current people, total unique people, and motion level

## Plot Results

```bash
python plot_people_log.py --log log.txt
```

To save the plot:

```bash
python plot_people_log.py --log log.txt --output people_plot.png
```

## Optional Evaluation

This repository does not include a labeled validation file by default, so metrics are optional instead of fake.

To evaluate predictions:

1. Copy `evaluation_labels_template.csv`
2. Fill it with two columns: `y_true` and `y_pred`
3. Run:

```bash
python evaluate_predictions.py --csv evaluation_labels.csv
```

## Notes for GitHub Submission

- Do not push the full dataset folder to GitHub.
- Do not push generated outputs unless you want a small sample demo.
- If you submit this academically, keep the project story consistent:
  this repo is a surveillance / people tracking project, not accident severity classification.
- Add 2-3 screenshots or a short GIF in the final GitHub version for stronger presentation.
- Add demo media inside the `assets/` folder before final sharing.

## GitHub Polish Added

- Reusable Python scripts for running, plotting, and evaluation
- Optional evaluation template instead of fake metrics
- GitHub issue templates and PR template
- Lightweight GitHub Actions syntax check

## Suggested GitHub Push Flow

```bash
git init
git add .
git commit -m "Initial smart surveillance project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Author

MAYANK PANDEY

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
