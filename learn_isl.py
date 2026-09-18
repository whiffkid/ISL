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

from utils import mediapipe_detection, draw_styled_landmarks, landmarks_data, is_hand_active, TextToSpeechEngine
from models import get_actions
from fingerspelling import ISL_ALPHABET, ISLFingerspellingEngine
from isl_learning_guide import evaluate_hand_posture, draw_tutor_camera_overlay, ISL_SYMBOL_GUIDES


def compute_sequence_similarity(seq1, seq2):
    """Compute cosine similarity and joint distance between two (30, 150) landmark sequences."""
    s1 = np.array(seq1, dtype=np.float32)
    s2 = np.array(seq2, dtype=np.float32)

    # Flattened cosine similarity
    dot = np.sum(s1 * s2)
    norm1 = np.linalg.norm(s1)
    norm2 = np.linalg.norm(s2)
    if norm1 > 1e-4 and norm2 > 1e-4:
        cos_sim = dot / (norm1 * norm2)
    else:
        cos_sim = 0.0

    # Pose & Hand distance metric
    dist = np.mean(np.linalg.norm(s1 - s2, axis=-1))
    score = max(0.0, min(100.0, (cos_sim * 0.65 + max(0, 1.0 - dist * 0.5) * 0.35) * 100.0))
    return score


def load_reference_sign(sign_name, data_dir="keypoint_data"):
    """Load average reference template sequence for a given sign."""
    target_dir = os.path.join(data_dir, sign_name.strip().lower().replace(" ", "_"))
    if not os.path.exists(target_dir):
        # Check standard name without underscore
        target_dir = os.path.join(data_dir, sign_name.strip().lower())
    
    if os.path.exists(target_dir):
        npy_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith('.npy')]
        if npy_files:
            valid_seqs = []
            for f in npy_files[:10]:
                try:
                    data = np.load(f)
                    if data.shape == (30, 150):
                        valid_seqs.append(data)
                except Exception:
                    pass
            if valid_seqs:
                return np.mean(valid_seqs, axis=0)
    return None


def run_isl_tutor(target_sign=None, data_dir="keypoint_data", camera_idx=0):
    actions = get_actions(data_dir)
    if not actions:
        actions = ["hello", "thank you", "good morning", "how are you", "pleased"]

    letters = [c for c in ISL_ALPHABET if len(c) == 1]
    current_idx = 0
    practice_mode = "sign"

    if target_sign:
        target_clean = target_sign.strip()
        if target_clean.upper() in letters or target_clean.upper() in ISL_SYMBOL_GUIDES:
            practice_mode = "alphabet"
            current_target = target_clean.upper() if target_clean.upper() in letters else target_clean
            if current_target in letters:
                current_idx = letters.index(current_target)
        elif target_clean.lower() in actions:
            practice_mode = "sign"
            current_target = target_clean.lower()
            current_idx = actions.index(current_target)
        else:
            current_target = actions[0]
    else:
        current_target = actions[0]

    tts = TextToSpeechEngine()
    spell_engine = ISLFingerspellingEngine()

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_idx}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    user_sequence = []
    score_history = []
    consecutive_high_scores = 0
    stars_earned = 0

    print("\n=======================================================")
    print("🎮 Interactive ISL Learning & Practice Tutor")
    print("=======================================================")
    print("Controls:\n"
          "  [n] : Next Sign / Letter\n"
          "  [p] : Previous Sign / Letter\n"
          "  [m] : Toggle Mode (Full Signs ↔ Alphabet A-Z)\n"
          "  [q] : Quit Tutor\n")

    if current_target in ISL_SYMBOL_GUIDES:
        g = ISL_SYMBOL_GUIDES[current_target]
        print(f"📖 Finger Guide for '{current_target.upper()}': {g['summary']}")
        print(f"🎯 Target Touch: {g.get('target_touch', 'N/A')}")
        print("Steps:")
        for s in g.get('steps', []):
            print(f"  {s}")
        print("=======================================================\n")

    ref_template = load_reference_sign(current_target, data_dir) if practice_mode == "sign" else None

    with mp_holistic.Holistic(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            h, w, _ = image.shape

            hand_active = is_hand_active(results)
            keypoints = landmarks_data(results)
            user_sequence.append(keypoints)
            user_sequence = user_sequence[-30:]

            current_score = 0.0
            feedback_tip = "Position hands in camera view"

            if practice_mode == "sign":
                if ref_template is not None and len(user_sequence) == 30 and hand_active:
                    current_score = compute_sequence_similarity(user_sequence, ref_template)
                    score_history.append(current_score)
                    if len(score_history) > 10:
                        score_history.pop(0)
                    
                    avg_score = np.mean(score_history)
                    if avg_score >= 82.0:
                        feedback_tip = "🌟 PERFECT FORM! Hold it!"
                        consecutive_high_scores += 1
                        if consecutive_high_scores == 25:
                            stars_earned += 1
                            tts.speak("Great job! Perfect sign!", lang="en")
                    elif avg_score >= 60.0:
                        feedback_tip = "👍 Good form! Keep steady."
                        consecutive_high_scores = 0
                    else:
                        feedback_tip = "💪 Match gesture rhythm and hand height."
                        consecutive_high_scores = 0
                else:
                    consecutive_high_scores = 0
            else:
                # Alphabet & Hand Signal practice with granular anatomical coaching
                eval_res = evaluate_hand_posture(results, current_target)
                current_score = eval_res['accuracy']
                feedback_tip = eval_res['primary_hint']

                # Draw augmented targeting crosshair & pointer connection on camera frame
                t_coords = eval_res.get('target_touch_coords')
                p_coords = eval_res.get('pointer_coords')
                if t_coords is not None:
                    tx, ty = int(t_coords[0] * w), int(t_coords[1] * h)
                    cv2.circle(image, (tx, ty), 16, (0, 215, 255), 2)
                    cv2.circle(image, (tx, ty), 6, (0, 165, 255), -1)
                    cv2.putText(image, "TARGET", (tx + 12, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)
                    if p_coords is not None:
                        px, py = int(p_coords[0] * w), int(p_coords[1] * h)
                        cv2.circle(image, (px, py), 8, (255, 105, 180), -1)
                        cv2.line(image, (px, py), (tx, ty), (255, 200, 50), 1, cv2.LINE_AA)

                if current_score >= 80.0:
                    consecutive_high_scores += 1
                    if consecutive_high_scores == 20:
                        stars_earned += 1
                        tts.speak(f"Awesome! Letter {current_target} completed!", lang="en")
                else:
                    consecutive_high_scores = 0

            # UI HUD Rendering
            # Top Banner
            cv2.rectangle(image, (0, 0), (w, 75), (25, 25, 25), -1)
            cv2.line(image, (0, 75), (w, 75), (70, 70, 70), 1)

            mode_label = "SIGN PRACTICE" if practice_mode == "sign" else "ALPHABET PRACTICE"
            cv2.putText(image, f"ISL TUTOR: {mode_label}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
            cv2.putText(image, f"TARGET: '{current_target.upper()}'", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

            # Star Badge on Top Right
            star_text = "⭐" * min(5, stars_earned) if stars_earned > 0 else "⭐ 0"
            cv2.putText(image, f"STARS: {stars_earned}", (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)
            cv2.putText(image, f"[n]Next  [p]Prev", (w - 180, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            # Accuracy Progress Bar (Bottom)
            bar_w = int((current_score / 100.0) * (w - 30))
            bar_color = (0, 230, 80) if current_score >= 80 else ((0, 165, 255) if current_score >= 50 else (0, 0, 240))
            cv2.rectangle(image, (15, h - 65), (w - 15, h - 45), (40, 40, 40), -1)
            cv2.rectangle(image, (15, h - 65), (15 + bar_w, h - 45), bar_color, -1)

            # Score & Feedback Text
            cv2.putText(image, f"ACCURACY: {int(current_score)}% | {feedback_tip}",
                        (15, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

            cv2.putText(image, "[n] Next Sign  [p] Previous  [m] Toggle Mode  [q] Quit",
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

            cv2.imshow("Interactive ISL Learning Tutor", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                if practice_mode == "sign":
                    current_idx = (current_idx + 1) % len(actions)
                    current_target = actions[current_idx]
                    ref_template = load_reference_sign(current_target, data_dir)
                else:
                    letters = [c for c in ISL_ALPHABET if len(c) == 1]
                    current_idx = (current_idx + 1) % len(letters)
                    current_target = letters[current_idx]
                score_history.clear()
                consecutive_high_scores = 0
                tts.speak(f"Next: {current_target}", lang="en")
            elif key == ord('p'):
                if practice_mode == "sign":
                    current_idx = (current_idx - 1) % len(actions)
                    current_target = actions[current_idx]
                    ref_template = load_reference_sign(current_target, data_dir)
                else:
                    letters = [c for c in ISL_ALPHABET if len(c) == 1]
                    current_idx = (current_idx - 1) % len(letters)
                    current_target = letters[current_idx]
                score_history.clear()
                consecutive_high_scores = 0
                tts.speak(f"Previous: {current_target}", lang="en")
            elif key == ord('m'):
                practice_mode = "alphabet" if practice_mode == "sign" else "sign"
                current_idx = 0
                if practice_mode == "sign":
                    current_target = actions[0]
                    ref_template = load_reference_sign(current_target, data_dir)
                else:
                    letters = [c for c in ISL_ALPHABET if len(c) == 1]
                    current_target = letters[0]
                score_history.clear()
                consecutive_high_scores = 0
                print(f"Switched Tutor Mode to: {practice_mode.upper()}")

    tts.stop()
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Gamified Interactive ISL Learning & Practice Tutor")
    parser.add_argument("--sign", type=str, help="Specific sign to practice (e.g. 'hello', 'thank you')")
    parser.add_argument("--data_dir", type=str, default="keypoint_data", help="Keypoint dataset directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    args = parser.parse_args()

    run_isl_tutor(target_sign=args.sign, data_dir=args.data_dir, camera_idx=args.camera)


if __name__ == "__main__":
    main()
