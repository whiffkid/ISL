import os
import sys

# Auto-reexec into local virtual environment if dependencies are not in current Python
try:
    import tensorflow
except ImportError:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.realpath(sys.executable) != os.path.realpath(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from models import load_model, get_actions
from utils import get_data, evaluate_detailed


def plot_training_history(history, save_path):
    """Plot and save training & validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy subplot
    ax1.plot(history.history.get('categorical_accuracy', []), label='Train Accuracy', color='#2563eb', lw=2)
    ax1.plot(history.history.get('val_categorical_accuracy', []), label='Val Accuracy', color='#16a34a', lw=2)
    ax1.set_title('Model Accuracy vs Epochs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Loss subplot
    ax2.plot(history.history.get('loss', []), label='Train Loss', color='#dc2626', lw=2)
    ax2.plot(history.history.get('val_loss', []), label='Val Loss', color='#ea580c', lw=2)
    ax2.set_title('Model Loss vs Epochs', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training curves to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Real-Time ISL Translation Model")
    parser.add_argument("--model", type=str, default="bilstm_attention",
                        choices=["lstm_v1", "lstm_v2", "lstm_v3", "bilstm_attention", "transformer"],
                        help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--data_dir", type=str, default="keypoint_data", help="Directory containing keypoint .npy data")
    parser.add_argument("--augment", action="store_true", default=True, help="Enable keypoint data augmentation")
    parser.add_argument("--augment_factor", type=int, default=2, help="Augmentation multiplier")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    device = '/GPU:0' if gpus else '/CPU:0'
    print(f"Using compute device: {device} (Available GPUs: {len(gpus)})")

    actions = get_actions(args.data_dir)
    print(f"Detected {len(actions)} actions: {actions}")

    print(f"Loading data from '{args.data_dir}' (Augment={args.augment}, Factor={args.augment_factor})...")
    X_train, X_test, y_train, y_test = get_data(
        train=True, test=True,
        data_folder=args.data_dir,
        augment=args.augment,
        augment_factor=args.augment_factor
    )
    print(f"Dataset shapes: X_train={X_train.shape}, y_train={y_train.shape} | X_test={X_test.shape}, y_test={y_test.shape}")

    model = load_model(args.model, pretrained=False, device=device, num_classes=len(actions))
    model.summary()

    save_dir = os.path.join("models", args.model)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "isl_model.keras")

    # Save class list metadata
    classes_file = os.path.join(save_dir, "classes.json")
    with open(classes_file, "w") as f:
        json.dump(actions, f, indent=2)
    print(f"Saved class metadata ({len(actions)} classes) to: {classes_file}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_categorical_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            verbose=1,
            mode='min',
            min_delta=0.0001,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy',
            patience=25,
            restore_best_weights=True,
            verbose=1
        )
    ]

    with tf.device(device):
        history = model.fit(
            X_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )

    # Save training visualization
    curves_path = os.path.join(save_dir, "training_curves.png")
    plot_training_history(history, curves_path)

    # Final detailed evaluation on test set
    acc, report, cm = evaluate_detailed(model, X_test, y_test, actions)
    print(f"\n=======================================================")
    print(f"Final Test Accuracy for {args.model}: {acc * 100:.2f}%")
    print(f"=======================================================")
    print("\nClassification Report:\n", report)

    # Save model in .h5 format as well for backwards compatibility
    h5_path = os.path.join(save_dir, "isl_model.h5")
    try:
        model.save(h5_path)
    except Exception:
        pass

    print(f"\nTraining completed! Model artifacts saved to '{save_dir}'.")


if __name__ == "__main__":
    main()
