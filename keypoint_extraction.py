import os
import cv2
import mediapipe as mp
import numpy as np
import argparse
from utils import mediapipe_detection, landmarks_data, pad_sequence

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

VALID_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def save_data(action, video_file, export_path, import_path, max_frame_length=30, skip_frame=2):
    MAX_FRAME_LENGTH = max_frame_length
    EXPORT_PATH = export_path
    IMPORT_PATH = import_path
    SKIP_FRAME = skip_frame

    frame_count = 0
    processed = 0
    data_per_video = []

    video_path = os.path.join(IMPORT_PATH, action, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Warning: Could not open video file {video_path}")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1

            if ret and (frame_count % SKIP_FRAME == 0):
                image, results = mediapipe_detection(frame, holistic)
                processed += 1
                data_per_video.append(landmarks_data(results))

            elif ret is False or processed == MAX_FRAME_LENGTH:
                if processed != MAX_FRAME_LENGTH:
                    data_per_video = pad_sequence(data_per_video, MAX_FRAME_LENGTH)
                
                data_per_video = np.array(data_per_video, dtype=np.float32)
                os.makedirs(os.path.join(EXPORT_PATH, action), exist_ok=True)
                
                base_name = os.path.splitext(video_file)[0]
                npy_path = os.path.join(EXPORT_PATH, action, f"{base_name}_skip_{skip_frame}.npy")
                
                print(f"Action: {action} | Video: {video_file}\n"
                      f"Processed Frames: {processed} | Data Shape: {data_per_video.shape}\n"
                      f"Saved to: {npy_path}\n"
                      f"---------------------------------------------")
                np.save(npy_path, data_per_video)
                break

        cap.release()


def main():
    parser = argparse.ArgumentParser(description="Extract ISL keypoints from video files")
    parser.add_argument("--import_path", type=str, default="greetings_data", help="Directory containing raw ISL action subfolders with videos")
    parser.add_argument("--export_path", type=str, default="keypoint_data", help="Directory to save extracted .npy keypoint data")
    parser.add_argument("--max_frames", type=int, default=30, help="Number of frames per video sequence")
    parser.add_argument("--skip_frame", type=int, default=2, help="Frame skipping step size")
    args = parser.parse_args()

    if not os.path.exists(args.import_path):
        print(f"Error: Source folder '{args.import_path}' not found.\n"
              f"Please download the dataset (e.g. INCLUDE Greetings dataset) and place action subfolders inside '{args.import_path}/'.")
        return

    actions = sorted([d for d in os.listdir(args.import_path) if os.path.isdir(os.path.join(args.import_path, d)) and not d.startswith('.')])
    if not actions:
        print(f"Error: No action subdirectories found in '{args.import_path}'.")
        return

    print(f"Extracting keypoints for actions ({len(actions)}): {actions}")

    for action in actions:
        action_dir = os.path.join(args.import_path, action)
        video_files = [f for f in os.listdir(action_dir) if f.lower().endswith(VALID_EXTENSIONS)]
        
        for video_file in video_files:
            save_data(action, video_file, args.export_path, args.import_path, args.max_frames, args.skip_frame)


if __name__ == "__main__":
    main()
