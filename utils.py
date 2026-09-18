import glob
import os
import sys
import ctypes
import queue
import threading
import time
import tempfile
import datetime

# Ensure matching local venv site-packages are accessible
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
for venv_site in glob.glob(os.path.join(base_dir, "venv*", "lib*", py_ver, "site-packages")):
    if venv_site not in sys.path and os.path.exists(venv_site):
        sys.path.insert(0, venv_site)

# Dynamic preloader for NVIDIA CUDA & cuDNN pip packages on Linux
for lib_dir in glob.glob(os.path.join(sys.prefix, 'lib*', 'python*', 'site-packages', 'nvidia', '*', 'lib')):
    for so_file in glob.glob(os.path.join(lib_dir, '*.so*')):
        try:
            ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            pass

# Compatibility fixes for MediaPipe solution_base & packet_getter with modern Protobuf (5.x+)
try:
    from google.protobuf import descriptor, symbol_database, message_factory
    if not hasattr(descriptor.FieldDescriptor, 'label'):
        try:
            descriptor.FieldDescriptor.label = property(lambda self: getattr(self, '_label', 1))
        except Exception:
            pass
    if not hasattr(symbol_database.SymbolDatabase, 'GetPrototype'):
        try:
            symbol_database.SymbolDatabase.GetPrototype = lambda self, desc: getattr(message_factory, 'GetMessageClass', lambda d: None)(desc)
        except Exception:
            pass
    if hasattr(message_factory, 'MessageFactory') and not hasattr(message_factory.MessageFactory, 'GetPrototype'):
        try:
            message_factory.MessageFactory.GetPrototype = lambda self, desc: getattr(message_factory, 'GetMessageClass', lambda d: None)(desc)
        except Exception:
            pass
except Exception:
    pass

import cv2
import numpy as np
import mediapipe as mp
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def to_categorical(y, num_classes=None):
    """Pure NumPy one-hot encoder to eliminate hard tensorflow dependency in utils."""
    y = np.array(y, dtype=int)
    if num_classes is None:
        num_classes = np.max(y) + 1
    return np.eye(num_classes, dtype=np.float32)[y]

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

BAR_COLORS = [
    (245, 117, 16), (117, 245, 16), (16, 117, 245),
    (255, 190, 11), (251, 86, 7), (255, 0, 110),
    (131, 56, 236), (58, 134, 255), (0, 245, 212)
]

EMERGENCY_WORDS = {
    "help", "doctor", "hospital", "police", "danger", "pain",
    "emergency", "ambulance", "fire", "accident", "bleed", "hurt"
}


class TextToSpeechEngine:
    """
    Asynchronous, non-blocking Multi-Lingual Text-To-Speech worker thread.
    Supports English (via fast pyttsx3 / spd-say) and Indian Regional Languages
    (Hindi, Tamil, Telugu, Bengali, Marathi, etc. via gTTS / pygame audio).
    """
    def __init__(self, rate=150, volume=1.0, default_lang="en"):
        self.queue = queue.Queue()
        self.rate = rate
        self.volume = volume
        self.current_lang = default_lang
        self.running = True
        self.last_spoken = ""
        self.last_spoken_time = 0

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def set_language(self, lang_code):
        self.current_lang = lang_code.lower()

    def _worker(self):
        engine = None
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
        except Exception:
            engine = None

        while self.running:
            try:
                item = self.queue.get(timeout=0.2)
                if item is None:
                    break

                text, lang = item
                if not text:
                    self.queue.task_done()
                    continue

                if lang == "en":
                    if engine is not None:
                        try:
                            engine.say(text)
                            engine.runAndWait()
                        except Exception:
                            os.system(f'spd-say "{text}" 2>/dev/null || espeak "{text}" 2>/dev/null || true')
                    else:
                        os.system(f'spd-say "{text}" 2>/dev/null || espeak "{text}" 2>/dev/null || true')
                else:
                    # Multi-lingual TTS using gTTS
                    try:
                        from gtts import gTTS
                        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
                            temp_audio = fp.name
                        
                        tts = gTTS(text=text, lang=lang, slow=False)
                        tts.save(temp_audio)

                        # Play via ffplay, mpv, or aplay
                        ret = os.system(f'mpv --no-video --really-quiet "{temp_audio}" 2>/dev/null || ffplay -nodisp -autoexit -loglevel quiet "{temp_audio}" 2>/dev/null || true')
                        try:
                            os.remove(temp_audio)
                        except Exception:
                            pass
                    except Exception:
                        # Fallback to local eSpeak with voice
                        os.system(f'espeak-ng -v {lang} "{text}" 2>/dev/null || true')

                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                pass

    def speak(self, text, lang=None, cooldown=1.5):
        now = time.time()
        target_lang = lang or self.current_lang
        key = f"{target_lang}:{text}"
        if text and (key != self.last_spoken or (now - self.last_spoken_time) > cooldown):
            self.last_spoken = key
            self.last_spoken_time = now
            self.queue.put((text, target_lang))

    def stop(self):
        self.running = False
        self.queue.put(None)


class TranscriptRecorder:
    """
    Session Subtitle (.srt) and Text Transcript Recorder.
    Captures live translated ISL sequences with precise timestamps.
    """
    def __init__(self):
        self.session_start = time.time()
        self.records = []  # list of (start_time, end_time, glosses_str, translation_str)
        self.current_entry_start = None

    def add_entry(self, glosses, translated_sentence):
        if not translated_sentence:
            return
        now = time.time() - self.session_start
        start_t = self.current_entry_start if self.current_entry_start is not None else max(0, now - 2.0)
        end_t = now + 1.0
        self.records.append((start_t, end_t, " ".join(glosses), translated_sentence))
        self.current_entry_start = now

    def _format_srt_time(self, seconds):
        td = datetime.timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def export_srt(self, output_path="transcripts/isl_session.srt"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, (st, et, glosses, text) in enumerate(self.records, 1):
                f.write(f"{i}\n")
                f.write(f"{self._format_srt_time(st)} --> {self._format_srt_time(et)}\n")
                f.write(f"{text}\n\n")
        return output_path

    def export_txt(self, output_path="transcripts/isl_session.txt"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"=== ISL Translation Session Transcript ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")
            for st, et, glosses, text in self.records:
                timestamp = f"[{int(st // 60):02d}:{int(st % 60):02d}]"
                f.write(f"{timestamp} ISL: [{glosses}] -> {text}\n")
        return output_path


def is_emergency_sign(sign_name):
    """Check if the recognized sign is an emergency / distress word."""
    if not sign_name:
        return False
    return sign_name.strip().lower() in EMERGENCY_WORDS


def mediapipe_detection(image, model):
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    else:
        return None, None


class FastCameraStream:
    """Threaded webcam reader for zero-latency, buffer-free 30-60fps frame capture."""
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        is_int_cam = isinstance(src, int) or (isinstance(src, str) and src.isdigit())
        if is_int_cam:
            self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.stream.set(cv2.CAP_PROP_FPS, 30)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stopped = True
                break
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def isOpened(self):
        return self.stream.isOpened() and not self.stopped

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.stream.release()


def draw_styled_landmarks(image, results, show_face=False):
    """
    Renders sleek, clean glowing joints for hands and upper body
    without cluttered dense face mesh overlays.
    """
    if image is None or results is None:
        return

    # Subtle face contours (optional)
    if show_face and results.face_landmarks:
        face_conns = getattr(mp_holistic, 'FACEMESH_CONTOURS', getattr(mp_holistic, 'FACE_CONNECTIONS', None))
        if face_conns:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, face_conns,
                mp_drawing.DrawingSpec(color=(120, 200, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 180, 255), thickness=1, circle_radius=1)
            )

    # Upper Body & Arm connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(230, 230, 230), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=1)
        )

    # Left Hand (Electric Cyan & Aqua)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 200), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
        )

    # Right Hand (Coral Neon & Amber)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 140, 255), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
        )


def landmarks_data(results, left_handed=False, normalize=False):
    """
    Extract 150-dim feature vector (33 pose*2 + 21 LH*2 + 21 RH*2).
    Supports left-handed mirroring and body-relative coordinate normalization.
    """
    # Pose coords
    if results and results.pose_landmarks:
        pose_arr = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark])
    else:
        pose_arr = np.zeros((33, 2))

    # Left Hand coords
    if results and results.left_hand_landmarks:
        lh_arr = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark])
    else:
        lh_arr = np.zeros((21, 2))

    # Right Hand coords
    if results and results.right_hand_landmarks:
        rh_arr = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark])
    else:
        rh_arr = np.zeros((21, 2))

    # Left-handed mode: Mirror horizontal X axis and swap left/right hands
    if left_handed:
        if results and results.pose_landmarks:
            pose_arr[:, 0] = 1.0 - pose_arr[:, 0]
        if results and results.left_hand_landmarks:
            lh_arr[:, 0] = 1.0 - lh_arr[:, 0]
        if results and results.right_hand_landmarks:
            rh_arr[:, 0] = 1.0 - rh_arr[:, 0]
        lh_arr, rh_arr = rh_arr, lh_arr

    # Body-relative normalization: center at midpoint of shoulders (landmarks 11 & 12)
    if normalize and results and results.pose_landmarks:
        l_sh = pose_arr[11]
        r_sh = pose_arr[12]
        mid_shoulder = (l_sh + r_sh) / 2.0
        sh_dist = np.linalg.norm(l_sh - r_sh)
        if sh_dist > 1e-3:
            pose_arr = (pose_arr - mid_shoulder) / sh_dist
            if results.left_hand_landmarks:
                lh_arr = (lh_arr - mid_shoulder) / sh_dist
            if results.right_hand_landmarks:
                rh_arr = (rh_arr - mid_shoulder) / sh_dist

    return np.concatenate([pose_arr.flatten(), lh_arr.flatten(), rh_arr.flatten()])


def is_hand_active(results):
    """Detect if signer is actively performing hand gestures or resting."""
    if not results:
        return False
    if results.left_hand_landmarks or results.right_hand_landmarks:
        return True
    
    # Or check if wrists in pose landmarks are raised above hips
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        left_wrist_y = lm[15].y if len(lm) > 15 else 1.0
        right_wrist_y = lm[16].y if len(lm) > 16 else 1.0
        left_hip_y = lm[23].y if len(lm) > 23 else 1.0
        right_hip_y = lm[24].y if len(lm) > 24 else 1.0
        avg_hip_y = (left_hip_y + right_hip_y) / 2.0
        
        if left_wrist_y < avg_hip_y or right_wrist_y < avg_hip_y:
            return True
            
    return False


def pad_sequence(data, max_frame_length=30):
    pose = np.zeros(33 * 2)
    lh = np.zeros(21 * 2)
    rh = np.zeros(21 * 2)

    padding = np.concatenate([pose, lh, rh])
    seq_length = len(data)

    if seq_length < max_frame_length:
        diff = max_frame_length - seq_length
        for _ in range(diff):
            data.append(padding)

    return data[:max_frame_length]


# -------------------------------------------------------------
# Data Augmentation Functions for Keypoint Sequences
# -------------------------------------------------------------

def augment_sequence(seq, jitter_std=0.012, scale_range=(0.92, 1.08), shift_range=(-0.04, 0.04), time_warp=True):
    """Apply spatial jitter, scaling, translation, and temporal warping to a (30, 150) sequence."""
    augmented = seq.copy()
    seq_len, feat_dim = augmented.shape

    # 1. Random spatial scaling
    scale = np.random.uniform(scale_range[0], scale_range[1])
    augmented = augmented * scale

    # 2. Random spatial shift/translation
    shift = np.random.uniform(shift_range[0], shift_range[1], size=(1, feat_dim))
    augmented = augmented + shift

    # 3. Gaussian noise jitter
    noise = np.random.normal(0, jitter_std, size=augmented.shape)
    augmented = augmented + noise

    # 4. Temporal Resampling / Warping
    if time_warp:
        target_len = np.random.randint(24, 36)
        orig_indices = np.linspace(0, seq_len - 1, num=target_len)
        resampled = np.zeros((target_len, feat_dim), dtype=np.float32)
        for i in range(feat_dim):
            resampled[:, i] = np.interp(orig_indices, np.arange(seq_len), augmented[:, i])
        
        final_seq = np.zeros((seq_len, feat_dim), dtype=np.float32)
        final_indices = np.linspace(0, target_len - 1, num=seq_len)
        for i in range(feat_dim):
            final_seq[:, i] = np.interp(final_indices, np.arange(target_len), resampled[:, i])
        augmented = final_seq

    # 5. Frame Dropout (zero out 1 or 2 random frames)
    if np.random.rand() > 0.5:
        drop_idx = np.random.randint(0, seq_len)
        augmented[drop_idx] = 0.0

    return np.clip(augmented, -3.0, 3.0).astype(np.float32)


def get_data(train=False, test=True, data_folder='keypoint_data', augment=False, augment_factor=2):
    sequences, labels = [], []

    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder '{data_folder}' does not exist. Please run keypoint_extraction.py first.")

    actions = sorted([d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d)) and not d.startswith('.')])
    if not actions:
        raise FileNotFoundError(f"No action subdirectories found in '{data_folder}'.")

    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        action_dir = os.path.join(data_folder, action)
        for video_file in os.listdir(action_dir):
            if video_file.endswith('.npy'):
                data = np.load(os.path.join(action_dir, video_file))
                sequences.append(data)
                labels.append(label_map[action])

    if not sequences:
        raise ValueError(f"No .npy sequence data files found in '{data_folder}'.")

    X = np.array(sequences, dtype=np.float32)
    y_raw = np.array(labels, dtype=np.int32)
    y = to_categorical(labels, num_classes=len(actions))

    # Stratified train-test split ensures each action class is represented in test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y_raw, random_state=42
    )

    if augment and train:
        aug_X, aug_y = [], []
        for _ in range(augment_factor):
            for i in range(len(X_train)):
                aug_X.append(augment_sequence(X_train[i]))
                aug_y.append(y_train[i])
        X_train = np.concatenate([X_train, np.array(aug_X, dtype=np.float32)], axis=0)
        y_train = np.concatenate([y_train, np.array(aug_y, dtype=np.float32)], axis=0)

    if train and not test:
        return X_train, y_train
    elif test and not train:
        return X_test, y_test
    return X_train, X_test, y_train, y_test


def accuracy(model, X_test, y_test):
    yhat = model.predict(X_test, verbose=0)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    return accuracy_score(ytrue, yhat)


def evaluate_detailed(model, X_test, y_test, actions):
    """Generate detailed classification report and confusion matrix."""
    yhat_probs = model.predict(X_test, verbose=0)
    ytrue = np.argmax(y_test, axis=1)
    yhat = np.argmax(yhat_probs, axis=1)

    labels_idx = list(range(len(actions)))
    acc = accuracy_score(ytrue, yhat)
    report = classification_report(ytrue, yhat, labels=labels_idx, target_names=actions, zero_division=0)
    cm = confusion_matrix(ytrue, yhat, labels=labels_idx)

    return acc, report, cm


def format_sentence(words):
    """Grammar formatting for translated sign sequence."""
    if not words:
        return ""
    text = " ".join(words).strip()
    return text.capitalize() + "."


def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    if res is not None and actions is not None:
        h, w, _ = output_frame.shape
        max_bars = min(len(actions), max(1, (h - 90) // 28))
        if len(actions) > max_bars:
            top_indices = np.argsort(res)[::-1][:max_bars]
        else:
            top_indices = list(range(len(actions)))

        for row, idx in enumerate(top_indices):
            if idx >= len(actions) or idx >= len(res):
                break
            prob = res[idx]
            color = BAR_COLORS[idx % len(BAR_COLORS)]
            bar_length = int(prob * 220)
            y_start = 65 + row * 28
            y_end = 88 + row * 28
            cv2.rectangle(output_frame, (0, y_start), (bar_length, y_end), color, -1)
            cv2.putText(output_frame, f"{actions[idx]} : {int(prob * 100)}%", (8, y_start + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2,
                        cv2.LINE_AA)
    return output_frame
