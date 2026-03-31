from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def run_surveillance(
    weights: str,
    source: int | str,
    output_path: Path,
    log_path: Path,
    conf_threshold: float,
    detection_interval: int,
    crowd_threshold: int,
    display: bool,
) -> None:
    model = YOLO(weights)
    tracker = DeepSort(max_age=20)
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    frame_width, frame_height = 480, 360
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 20

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    prev_frame = None
    frame_count = 0
    all_track_ids: set[int] = set()
    tracks = []

    with log_path.open("w", newline="") as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(["frame", "current_people", "total_unique_people", "motion_level"])

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            frame_count += 1
            detections = []

            if frame_count % detection_interval == 0:
                results = model(frame, verbose=False)

                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])

                        if cls == 0 and conf >= conf_threshold:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

                tracks = tracker.update_tracks(detections, frame=frame)
            else:
                tracks = tracker.update_tracks([], frame=frame)

            current_track_ids = set()

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = int(track.track_id)
                left, top, right, bottom = map(int, track.to_ltrb())
                current_track_ids.add(track_id)
                all_track_ids.add(track_id)

                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (left, max(top - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            current_people = len(current_track_ids)
            total_unique_people = len(all_track_ids)

            if prev_frame is not None and frame_count % detection_interval == 0:
                gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                gray1 = cv2.GaussianBlur(gray1, (7, 7), 0)
                gray2 = cv2.GaussianBlur(gray2, (7, 7), 0)

                diff = cv2.absdiff(gray1, gray2)
                _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5, 5), np.uint8)
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                motion_level = int(np.sum(thresh) / 255)

                if display:
                    cv2.imshow("Motion", thresh)
            else:
                motion_level = 0

            prev_frame = frame.copy()

            cv2.putText(frame, f"Current People: {current_people}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Total Unique People: {total_unique_people}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Motion: {motion_level}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            if current_people > crowd_threshold:
                cv2.putText(frame, "ALERT: Crowd Detected!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            csv_writer.writerow([frame_count, current_people, total_unique_people, motion_level])
            writer.write(frame)

            if display:
                cv2.imshow("Smart Surveillance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run YOLOv8 + DeepSORT smart surveillance.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to YOLO weights.")
    parser.add_argument("--source", default="0", help="Webcam index like 0, or a video file path.")
    parser.add_argument("--output", default="output.avi", help="Path to save annotated output video.")
    parser.add_argument("--log", default="log.txt", help="Path to save CSV log data.")
    parser.add_argument("--conf-threshold", type=float, default=0.4, help="Minimum confidence for person detections.")
    parser.add_argument("--detection-interval", type=int, default=5, help="Run YOLO once every N frames.")
    parser.add_argument("--crowd-threshold", type=int, default=3, help="Show alert when current people exceed this count.")
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow windows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_surveillance(
        weights=args.weights,
        source=parse_source(args.source),
        output_path=Path(args.output),
        log_path=Path(args.log),
        conf_threshold=args.conf_threshold,
        detection_interval=args.detection_interval,
        crowd_threshold=args.crowd_threshold,
        display=not args.no_display,
    )


if __name__ == "__main__":
    main()
