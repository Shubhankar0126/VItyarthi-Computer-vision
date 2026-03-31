from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_log(log_path: Path, output_path: Path | None) -> None:
    df = pd.read_csv(log_path)
    df["current_people_smooth"] = df["current_people"].rolling(window=3).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df["frame"], df["current_people"], label="Current People")
    plt.plot(df["frame"], df["total_unique_people"], label="Total Unique People")
    plt.plot(df["frame"], df["current_people_smooth"], label="Smoothed Current People")
    plt.xlabel("Frame Number")
    plt.ylabel("People Count")
    plt.title("People Count Over Time")
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot people-count trends from surveillance CSV logs.")
    parser.add_argument("--log", default="log.txt", help="CSV log file created by run_surveillance.py")
    parser.add_argument("--output", default="", help="Optional output image path")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    plot_log(Path(args.log), output)


if __name__ == "__main__":
    main()
