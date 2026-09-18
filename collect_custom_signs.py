import os
import sys

# Auto-reexec into local virtual environment if dependencies are not in current Python
try:
    import tensorflow
except ImportError:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if os.path.exists(venv_py) and os.path.realpath(sys.executable) != os.path.realpath(venv_py):
        os.execv(venv_py, [venv_py] + sys.argv)

import time
import argparse
import numpy as np
import cv2
import mediapipe as mp

from utils import mediapipe_detection, draw_styled_landmarks, landmarks_data, pad_sequence


def record_sign(sign_name, num_samples=15, sequence_length=30, countdown_secs=2, export_dir="keypoint_data", camera_idx=0):
    """
    Interactively record keypoint sequence samples for a new ISL sign via webcam.
    """
    clean_sign_name = sign_name.strip().lower().replace(" ", "_")
    target_dir = os.path.join(export_dir, clean_sign_name)
    os.makedirs(target_dir, exist_ok=True)

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"Error: Could not open camera device {camera_idx}.")
        return

    # Find existing sample count
    existing_files = [f for f in os.listdir(target_dir) if f.endswith(".npy")]
    start_sample_idx = len(existing_files)

    print(f"\n=======================================================")
    print(f"Recording New ISL Sign: '{clean_sign_name}'")
    print(f"Target Directory: {target_dir}")
    print(f"Samples to record: {num_samples} (Frames per sample: {sequence_length})")
    print(f"=======================================================")
    print("Press 'q' at any time to quit early.\n")

    with mp_holistic.Holistic(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for sample_num in range(start_sample_idx, start_sample_idx + num_samples):
            # Phase 1: Countdown before recording
            countdown_start = time.time()
            while time.time() - countdown_start < countdown_secs:
                ret, frame = cap.read()
                if not ret:
                    break

                remaining = countdown_secs - int(time.time() - countdown_start)
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                h, w, _ = image.shape
                # Overlay banner
                cv2.rectangle(image, (0, 0), (w, 60), (30, 30, 30), -1)
                cv2.putText(image, f"Sign: '{clean_sign_name.upper()}' | Sample {sample_num + 1}/{start_sample_idx + num_samples}",
                            (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Countdown prompt
                cv2.putText(image, f"Get Ready: {remaining}s", (w // 2 - 120, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 165, 255), 3)

                cv2.imshow("Record Custom ISL Sign", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Phase 2: Record sequence
            sequence_data = []
            print(f"Recording sample {sample_num + 1}...")

            for frame_idx in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                keypoints = landmarks_data(results)
                sequence_data.append(keypoints)

                h, w, _ = image.shape
                # Recording banner (RED)
                cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 180), -1)
                cv2.putText(image, f"RECORDING '{clean_sign_name.upper()}' [{frame_idx + 1}/{sequence_length}]",
                            (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                cv2.imshow("Record Custom ISL Sign", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # Pad / process sequence
            if len(sequence_data) < sequence_length:
                sequence_data = pad_sequence(sequence_data, sequence_length)
            
            sequence_np = np.array(sequence_data[:sequence_length], dtype=np.float32)
            out_file = os.path.join(target_dir, f"{clean_sign_name}_sample_{sample_num}.npy")
            np.save(out_file, sequence_np)
            print(f"  ✓ Saved: {out_file} (Shape: {sequence_np.shape})")

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n=======================================================")
    print(f"Done! Successfully recorded samples for '{clean_sign_name}'.")
    print(f"Total samples now available in {target_dir}: {len(os.listdir(target_dir))}")
    print(f"\nTo retrain model with your new sign, run:")
    print(f"  python train.py --model bilstm_attention --epochs 80 --augment")
    print(f"=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Record any custom ISL sign and extract skeletal keypoints via webcam")
    parser.add_argument("--sign", type=str, required=True, help="Name of the ISL sign to record (e.g. 'water', 'doctor', 'help')")
    parser.add_argument("--samples", type=int, default=15, help="Number of samples to record (default: 15)")
    parser.add_argument("--frames", type=int, default=30, help="Frames per sample sequence (default: 30)")
    parser.add_argument("--countdown", type=int, default=2, help="Countdown seconds before each sample (default: 2)")
    parser.add_argument("--export_dir", type=str, default="keypoint_data", help="Output directory for keypoint .npy files")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    args = parser.parse_args()

    record_sign(
        sign_name=args.sign,
        num_samples=args.samples,
        sequence_length=args.frames,
        countdown_secs=args.countdown,
        export_dir=args.export_dir,
        camera_idx=args.camera
    )


if __name__ == "__main__":
    main()
