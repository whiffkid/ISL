import os
import sys

# Auto-reexec into local virtual environment if dependencies are not in current Python
try:
    import tensorflow
except ImportError:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.realpath(sys.executable) != os.path.realpath(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)

import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import get_data, evaluate_detailed
from models import load_model, get_actions


def plot_confusion_matrix(cm, classes, save_path):
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)

    # Annotate numbers in cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > (cm.max() / 2) else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_xlabel('Predicted Gesture', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Gesture', fontsize=11, fontweight='bold')
    ax.set_title('ISL Gesture Recognition Confusion Matrix', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ISL Translation Model")
    parser.add_argument("--model", type=str, default="lstm_v3",
                        choices=["lstm_v1", "lstm_v2", "lstm_v3", "bilstm_attention", "transformer"],
                        help="Model architecture to evaluate")
    parser.add_argument("--data_dir", type=str, default="keypoint_data", help="Directory containing keypoint .npy data")
    parser.add_argument("--plot", action="store_true", default=True, help="Save confusion matrix heatmap plot")
    args = parser.parse_args()

    actions = get_actions(args.data_dir)
    print(f"Evaluating Model: '{args.model}' across {len(actions)} classes")

    try:
        model = load_model(args.model, pretrained=True, training=False, num_classes=len(actions))
    except FileNotFoundError as e:
        print(f"Pretrained model file not found: {e}")
        print("Falling back to evaluating an un-trained model structure...")
        model = load_model(args.model, pretrained=False, training=False, num_classes=len(actions))

    model.summary()

    try:
        X_test, y_test = get_data(train=False, test=True, data_folder=args.data_dir)
        acc, report, cm = evaluate_detailed(model, X_test, y_test, actions)

        print(f"\n=======================================================")
        print(f"Model Evaluation Accuracy on Test Set: {acc * 100:.2f}%")
        print(f"=======================================================\n")
        print("Classification Report:")
        print(report)

        print("\nConfusion Matrix:")
        print(cm)

        if args.plot:
            save_dir = os.path.join("models", args.model)
            os.makedirs(save_dir, exist_ok=True)
            cm_plot_path = os.path.join(save_dir, "confusion_matrix.png")
            plot_confusion_matrix(cm, actions, cm_plot_path)

    except Exception as e:
        print(f"\nCould not evaluate model on '{args.data_dir}': {e}")


if __name__ == "__main__":
    main()
