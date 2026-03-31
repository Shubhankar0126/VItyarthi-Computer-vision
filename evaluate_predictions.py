from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate(csv_path: Path, true_column: str, pred_column: str, figure_path: Path | None) -> None:
    df = pd.read_csv(csv_path)

    if true_column not in df.columns or pred_column not in df.columns:
        raise ValueError(f"CSV must contain '{true_column}' and '{pred_column}' columns.")

    y_true = df[true_column]
    y_pred = df[pred_column]

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if figure_path is not None:
        plt.savefig(figure_path)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions from a labeled CSV file.")
    parser.add_argument("--csv", default="evaluation_labels.csv", help="CSV file with y_true and y_pred columns.")
    parser.add_argument("--true-column", default="y_true", help="Ground-truth label column name.")
    parser.add_argument("--pred-column", default="y_pred", help="Prediction column name.")
    parser.add_argument("--figure", default="", help="Optional output path for confusion matrix image.")
    args = parser.parse_args()

    figure_path = Path(args.figure) if args.figure else None
    evaluate(Path(args.csv), args.true_column, args.pred_column, figure_path)


if __name__ == "__main__":
    main()
