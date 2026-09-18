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

# Set Qt platform fallback & suppress font search warnings for OpenCV GUI on Linux
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")

from utils import (
    mediapipe_detection,
    draw_styled_landmarks,
    landmarks_data,
    is_hand_active,
    prob_viz,
    format_sentence,
    TextToSpeechEngine,
    TranscriptRecorder,
    is_emergency_sign,
    FastCameraStream
)
from models import load_model, get_actions
from fingerspelling import ISLFingerspellingEngine
from grammar_translator import ISLGrammarTranslator
import mediapipe as mp


def main():
    parser = argparse.ArgumentParser(description="Universal Real-Time ISL Translation (Dynamic Signs, Fingerspelling & Multilingual Grammar)")
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: '0' for default webcam, or path to video file (.mp4/.avi/.mov)")
    parser.add_argument("--model", type=str, default="bilstm_attention",
                        choices=["bilstm_attention", "lstm_v3", "lstm_v1", "lstm_v2", "transformer"],
                        help="Neural network model architecture to use for sign classification")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["sign", "spell", "hybrid"],
                        help="Translation mode: 'sign' (full gestures), 'spell' (manual alphabet), 'hybrid' (both)")
    parser.add_argument("--lang", type=str, default="en",
                        choices=["en", "hi", "ta", "te", "bn", "mr", "kn", "ml", "gu", "pa"],
                        help="Target output language for translation and speech (en, hi, ta, te, etc.)")
    parser.add_argument("--left_handed", action="store_true", help="Enable Left-Handed Signer mirroring mode")
    parser.add_argument("--mirror", action="store_true", default=False, help="Enable selfie mirror view for webcam")
    parser.add_argument("--face", action="store_true", default=False, help="Render subtle face contour tracking")
    parser.add_argument("--thresh", type=float, default=0.55, help="Confidence threshold for sign prediction")
    parser.add_argument("--complexity", type=int, default=0, choices=[0, 1, 2], help="MediaPipe model complexity (0=fastest/zero-lag)")
    parser.add_argument("--step", type=int, default=2, help="Run neural network prediction every N frames")
    parser.add_argument("--tts", action="store_true", default=True, help="Enable real-time Text-to-Speech audio output")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--learn", type=str, default=None,
                        help="Launch interactive learning tutor for a specific symbol or sign (e.g. --learn A or --learn hello)")
    args = parser.parse_args()

    is_webcam = (args.source == "0" or args.source == 0)
    source = 0 if is_webcam else args.source

    if args.learn:
        from learn_isl import run_isl_tutor
        camera_idx = source if isinstance(source, int) else 0
        run_isl_tutor(target_sign=args.learn, camera_idx=camera_idx)
        return

    actions = get_actions(model_name=args.model)
    print(f"Loaded {len(actions)} ISL actions: {actions}")

    print(f"Loading dynamic sign model '{args.model}'...")
    try:
        model = load_model(args.model, pretrained=True, training=False, num_classes=len(actions))
    except Exception as e:
        print(f"Could not load pretrained model: {e}")
        print("Loading un-trained model structure for demonstration...")
        model = load_model(args.model, pretrained=False, training=False, num_classes=len(actions))

    # Compile fast inference graph
    @tf.function(reduce_retracing=True)
    def fast_predict(x):
        return model(x, training=False)

    # Warm-up graph compilation for instant zero-latency inference
    _ = fast_predict(np.zeros((1, 30, 150), dtype=np.float32))

    # Initialize Fingerspelling, Grammar, and Transcript engines
    spell_engine = ISLFingerspellingEngine()
    grammar_engine = ISLGrammarTranslator(target_language=args.lang)
    transcript_recorder = TranscriptRecorder()

    # Initialize Multi-Lingual Text-To-Speech engine
    tts_engine = TextToSpeechEngine(default_lang=args.lang) if args.tts else None

    sequence = []
    rolling_preds = []
    pred_cooldown = 0
    sentence_words = []
    current_mode = args.mode  # "sign", "spell", "hybrid"
    current_lang = args.lang
    left_handed = args.left_handed
    res = None
    thresh = args.thresh
    frame_count = 0
    fps = 0.0
    prev_time = time.time()
    last_spell_char = ""
    last_translated_sentence = ""
    translated_text = ""
    last_gloss_key = None
    suggestions = []
    sos_active = False
    sos_start_time = 0

    LANGUAGES_CYCLE = ["en", "hi", "ta", "te", "bn", "mr", "kn", "ml", "gu", "pa"]

    mp_holistic = mp.solutions.holistic
    cap = FastCameraStream(source, width=args.width, height=args.height)

    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'.")
        return

    width = args.width
    height = args.height

    print(f"\n=======================================================")
    print(f"Universal ISL Real-Time Translation Stream Started ({width}x{height})")
    print(f"Active Mode: {current_mode.upper()} | Target Language: {current_lang.upper()} ({grammar_engine.SUPPORTED_LANGUAGES.get(current_lang, '')})")
    print(f"Signer Handedness: {'LEFT-HANDED (Mirrored)' if left_handed else 'RIGHT-HANDED'}")
    print("Controls:\n"
          "  [m] : Toggle Mode (SIGN / SPELL / HYBRID)\n"
          "  [l] : Cycle Language (EN -> HI -> TA -> TE -> BN -> MR -> KN -> ML)\n"
          "  [h] : Toggle Handedness (Right / Left-Handed)\n"
          "  [1/2/3] : Accept Fingerspelling Autocomplete Suggestion\n"
          "  [e] : Export Subtitles & Session Transcript (.srt / .txt)\n"
          "  [c] : Clear sentence\n"
          "  [b] : Backspace\n"
          "  [s] : Toggle TTS Voice\n"
          "  [t] : Speak translated sentence\n"
          "  [q] : Quit & Save Session")
    print(f"=======================================================\n")

    tts_enabled = args.tts

    with mp_holistic.Holistic(
        model_complexity=args.complexity,
        smooth_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=False
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_count += 1
            cur_time = time.time()
            dt = cur_time - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = cur_time

            # 1. MediaPipe Landmark Detection
            image, results = mediapipe_detection(frame, holistic)

            # Draw visual tracking skeleton on hands & body
            draw_styled_landmarks(image, results, show_face=args.face)

            # Check if signer is actively performing hand gestures
            hand_active = is_hand_active(results)

            # 2. Extract keypoints (with handedness mirroring if enabled) & append to sliding window
            keypoints = landmarks_data(results, left_handed=left_handed)
            sequence.append(keypoints)
            sequence = sequence[-60:]

            # 3A. Dynamic Sign Recognition (in SIGN or HYBRID mode)
            if current_mode in ["sign", "hybrid"]:
                pred_cooldown = max(0, pred_cooldown - 1)
                if len(sequence) == 60 and (frame_count % args.step == 0) and hand_active and pred_cooldown == 0:
                    # Sample every 2nd frame to match 30-frame SKIP_FRAME=2 training format
                    sampled_seq = np.array(sequence)[::2]
                    input_seq = np.expand_dims(sampled_seq, axis=0)
                    raw_preds = fast_predict(input_seq).numpy()[0]
                    rolling_preds.append(raw_preds)
                    if len(rolling_preds) > 3:
                        rolling_preds = rolling_preds[-3:]
                    res = np.mean(rolling_preds, axis=0)

                    max_idx = int(np.argmax(res))
                    confidence = res[max_idx]

                    if confidence > thresh and max_idx < len(actions):
                        pred_action = actions[max_idx]

                        if not sentence_words or pred_action != sentence_words[-1]:
                            sentence_words.append(pred_action)
                            
                            # Check emergency distress trigger
                            if is_emergency_sign(pred_action):
                                sos_active = True
                                sos_start_time = time.time()

                            if tts_enabled and tts_engine:
                                tts_engine.speak(pred_action, lang="en")

                        if len(sentence_words) > 8:
                            sentence_words = sentence_words[-8:]

                        # Debounce buffer on confirmed sign
                        sequence = sequence[-16:]
                        rolling_preds = []
                        pred_cooldown = 18

            # 3B. Fingerspelling Recognition (in SPELL or HYBRID mode)
            if current_mode in ["spell", "hybrid"] and hand_active:
                char, current_word, is_new_letter = spell_engine.update_stream(results)
                if char:
                    last_spell_char = char
                    if is_new_letter and tts_enabled and tts_engine and current_mode == "spell":
                        tts_engine.speak(char, lang="en", cooldown=0.4)
                
                # Fetch live dictionary suggestions only when word state changes
                if is_new_letter:
                    suggestions = spell_engine.get_word_suggestions(max_suggestions=3)
            elif not spell_engine.current_word:
                suggestions = []

            # 4. Render live probability bar graph for dynamic signs
            if current_mode in ["sign", "hybrid"] and res is not None:
                image = prob_viz(res, actions, image)

            # 5. Multilingual Grammar Translation (computed only on gloss buffer updates)
            curr_gloss_key = (tuple(sentence_words), spell_engine.current_word.strip(), current_lang)
            if curr_gloss_key != last_gloss_key:
                last_gloss_key = curr_gloss_key
                full_gloss_list = list(sentence_words)
                if spell_engine.current_word.strip():
                    full_gloss_list.append(spell_engine.current_word.strip())

                translated_text, _ = grammar_engine.translate(full_gloss_list, target_lang=current_lang)
                if translated_text and translated_text != last_translated_sentence:
                    last_translated_sentence = translated_text
                    transcript_recorder.add_entry(full_gloss_list, translated_text)

            # 6. Render Modern HUD UI
            # Top Banner for Translated Sentence
            cv2.rectangle(image, (0, 0), (width, 62), (20, 20, 20), -1)
            cv2.line(image, (0, 62), (width, 62), (80, 80, 80), 1)

            lang_label = f"{current_lang.upper()}:"
            cv2.putText(image, lang_label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(image, translated_text if translated_text else "Waiting for ISL signs...",
                        (68, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

            # Sub-row: Raw ISL gloss sequence & Fingerspelled word
            gloss_str = " ".join(sentence_words) if sentence_words else ""
            spell_str = f" | Spelled: '{spell_engine.current_word}'" if spell_engine.current_word else ""
            cv2.putText(image, f"ISL: [{gloss_str}]{spell_str}",
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 230, 255), 1, cv2.LINE_AA)

            # Mode & Language Badges on Top Right
            mode_colors = {"sign": (255, 140, 0), "spell": (255, 0, 200), "hybrid": (0, 230, 100)}
            mode_badge = f"{current_mode.upper()} | {current_lang.upper()}"
            cv2.putText(image, mode_badge, (width - 240, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, mode_colors.get(current_mode, (255, 255, 255)), 2, cv2.LINE_AA)

            status_text = "SIGNING" if hand_active else "IDLE"
            hand_badge = " [L-HAND]" if left_handed else ""
            status_color = (0, 255, 128) if hand_active else (150, 150, 150)
            cv2.putText(image, f"{status_text}{hand_badge} | FPS: {fps:.1f}", (width - 240, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)

            # Fingerspelling Suggestions Banner (if spelling)
            if (current_mode in ["spell", "hybrid"]) and spell_engine.current_word.strip() and suggestions:
                sug_text = "Suggestions: " + " | ".join([f"[{i+1}] {w}" for i, w in enumerate(suggestions)])
                cv2.rectangle(image, (0, 64), (width, 92), (35, 35, 35), -1)
                cv2.putText(image, sug_text, (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # Active Fingerspelled Char Box on right
            if (current_mode == "spell" or spell_engine.current_word) and last_spell_char:
                cv2.rectangle(image, (width - 110, 98), (width - 10, 168), (40, 40, 40), -1)
                cv2.rectangle(image, (width - 110, 98), (width - 10, 168), (0, 200, 255), 2)
                cv2.putText(image, "CHAR", (width - 95, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(image, last_spell_char, (width - 80, 156), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

            # Emergency SOS Flashing Banner (if active within last 4s)
            if sos_active:
                if time.time() - sos_start_time < 4.0:
                    blink = int(time.time() * 4) % 2 == 0
                    sos_bg = (0, 0, 220) if blink else (255, 255, 255)
                    sos_fg = (255, 255, 255) if blink else (0, 0, 220)
                    cv2.rectangle(image, (0, height - 80), (width, height - 35), sos_bg, -1)
                    cv2.putText(image, "🚨 EMERGENCY / DISTRESS SIGN DETECTED 🚨", (width // 2 - 220, height - 52),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, sos_fg, 2, cv2.LINE_AA)
                else:
                    sos_active = False

            # Controls hint at bottom
            tts_status = "ON" if tts_enabled else "OFF"
            cv2.putText(image, f"[m]Mode  [l]Lang:{current_lang.upper()}  [h]Handedness  [e]Export SRT  [c]Clear  [s]TTS:{tts_status}  [t]Speak  [q]Quit",
                        (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)

            # 7. Display frame & handle key events
            try:
                cv2.imshow('Universal Real-Time ISL Translation', image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    modes = ["sign", "spell", "hybrid"]
                    current_mode = modes[(modes.index(current_mode) + 1) % len(modes)]
                    print(f"Switched Mode to: {current_mode.upper()}")
                elif key == ord('l'):
                    next_idx = (LANGUAGES_CYCLE.index(current_lang) + 1) % len(LANGUAGES_CYCLE)
                    current_lang = LANGUAGES_CYCLE[next_idx]
                    grammar_engine.set_target_language(current_lang)
                    if tts_engine:
                        tts_engine.set_language(current_lang)
                    print(f"Switched Target Language to: {current_lang.upper()} ({grammar_engine.SUPPORTED_LANGUAGES.get(current_lang, '')})")
                elif key == ord('h'):
                    left_handed = not left_handed
                    print(f"Handedness Mode: {'LEFT-HANDED (Mirrored)' if left_handed else 'RIGHT-HANDED'}")
                elif key in [ord('1'), ord('2'), ord('3')]:
                    sug_idx = key - ord('1')
                    chosen = spell_engine.autocomplete(sug_idx)
                    print(f"Autocompleted: '{chosen}'")
                elif key == ord('e'):
                    ts = int(time.time())
                    srt_file = transcript_recorder.export_srt(f"transcripts/isl_session_{ts}.srt")
                    txt_file = transcript_recorder.export_txt(f"transcripts/isl_session_{ts}.txt")
                    print(f"\n✓ Session exported successfully:")
                    print(f"  Subtitles: {srt_file}")
                    print(f"  Transcript: {txt_file}\n")
                elif key == ord('c'):
                    sentence_words.clear()
                    spell_engine.clear()
                    print("Sentence buffer cleared.")
                elif key == ord('b'):
                    if spell_engine.current_word:
                        spell_engine.backspace()
                    elif sentence_words:
                        sentence_words.pop()
                    print("Backspace executed.")
                elif key == ord('s'):
                    tts_enabled = not tts_enabled
                    print(f"TTS Audio: {'Enabled' if tts_enabled else 'Disabled'}")
                elif key == ord('t'):
                    if tts_enabled and tts_engine and translated_text:
                        tts_engine.speak(translated_text, lang=current_lang, cooldown=0.0)
                        print(f"Speaking ({current_lang}): '{translated_text}'")
            except cv2.error:
                pass

    if tts_engine:
        tts_engine.stop()

    # Automatically save session transcript on exit
    if transcript_recorder.records:
        ts = int(time.time())
        srt_file = transcript_recorder.export_srt(f"transcripts/session_{ts}.srt")
        txt_file = transcript_recorder.export_txt(f"transcripts/session_{ts}.txt")
        print(f"\nSaved session transcript to: {srt_file} and {txt_file}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
