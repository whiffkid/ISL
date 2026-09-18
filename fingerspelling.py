import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib

# Standard Indian Sign Language (ISL) two-handed manual alphabet & digits
ISL_ALPHABET = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'SPACE', 'BACKSPACE'
]

FINGERSPELLING_MODEL_DIR = os.path.join("models", "fingerspelling")
MODEL_PATH = os.path.join(FINGERSPELLING_MODEL_DIR, "fingerspelling_model.joblib")
CLASSES_PATH = os.path.join(FINGERSPELLING_MODEL_DIR, "classes.json")

# High-frequency dictionary of English & ISL common words and names for autocomplete/autocorrect
COMMON_ISL_DICTIONARY = [
    "HELLO", "THANK", "YOU", "PLEASE", "WELCOME", "SORRY", "ALRIGHT", "GOOD", "MORNING",
    "AFTERNOON", "EVENING", "NIGHT", "NAME", "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO",
    "WATER", "FOOD", "HELP", "DOCTOR", "HOSPITAL", "POLICE", "AMBULANCE", "MEDICINE", "PAIN",
    "EMERGENCY", "FAMILY", "MOTHER", "FATHER", "BROTHER", "SISTER", "FRIEND", "TEACHER", "STUDENT",
    "SCHOOL", "COLLEGE", "OFFICE", "WORK", "HOME", "HOUSE", "ROOM", "BATHROOM", "TOILET",
    "TIME", "TODAY", "TOMORROW", "YESTERDAY", "NOW", "LATER", "MORNING", "EVENING", "NIGHT",
    "LOVE", "LIKE", "WANT", "NEED", "KNOW", "UNDERSTAND", "SEE", "HEAR", "SPEAK", "READ", "WRITE",
    "INDIA", "DELHI", "MUMBAI", "BANGALORE", "CHENNAI", "KOLKATA", "HYDERABAD", "KERALA",
    "HAPPY", "SAD", "TIRED", "HUNGRY", "THIRSTY", "BUSY", "FINE", "READY", "BEAUTIFUL",
    "RED", "BLUE", "GREEN", "YELLOW", "WHITE", "BLACK", "ORANGE", "PURPLE", "BROWN",
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN",
    "YES", "NO", "MAYBE", "AGAIN", "STOP", "START", "COME", "GO", "WAIT", "CALL", "SEND"
]


def get_hand_metrics(landmarks):
    """
    Computes finger extension flags, curl ratios, and key fingertip positions.
    Landmarks: list of 21 MediaPipe landmark objects or arrays.
    """
    if landmarks is None or len(landmarks) < 21:
        return None

    pts = np.array([[lm.x, lm.y] for lm in landmarks[:21]])
    wrist = pts[0]
    
    # Hand scale reference: distance from wrist (0) to middle MCP knuckle (9)
    span = np.linalg.norm(pts[9] - pts[0])
    if span < 1e-4:
        span = 1.0

    # Finger tip indices: Thumb=4, Index=8, Middle=12, Ring=16, Pinky=20
    # MCP knuckle indices: Thumb=2, Index=5, Middle=9, Ring=13, Pinky=17
    # PIP knuckle indices: Thumb=3, Index=6, Middle=10, Ring=14, Pinky=18
    # DIP knuckle indices: Thumb=3, Index=7, Middle=11, Ring=15, Pinky=19

    # A finger is extended if tip is further from wrist than PIP knuckle,
    # OR if tip is further from MCP knuckle than PIP knuckle is.
    index_ext = bool((np.linalg.norm(pts[8] - wrist) > np.linalg.norm(pts[6] - wrist) * 1.04) or
                     (np.linalg.norm(pts[8] - pts[5]) > np.linalg.norm(pts[6] - pts[5]) * 1.06))
    middle_ext = bool((np.linalg.norm(pts[12] - wrist) > np.linalg.norm(pts[10] - wrist) * 1.04) or
                      (np.linalg.norm(pts[12] - pts[9]) > np.linalg.norm(pts[10] - pts[9]) * 1.06))
    ring_ext = bool((np.linalg.norm(pts[16] - wrist) > np.linalg.norm(pts[14] - wrist) * 1.04) or
                    (np.linalg.norm(pts[16] - pts[13]) > np.linalg.norm(pts[14] - pts[13]) * 1.06))
    pinky_ext = bool((np.linalg.norm(pts[20] - wrist) > np.linalg.norm(pts[18] - wrist) * 1.04) or
                     (np.linalg.norm(pts[20] - pts[17]) > np.linalg.norm(pts[18] - pts[17]) * 1.06))

    # Thumb is extended if tip is outstretched away from index MCP and pinky MCP,
    # or pointing straight up (thumbs up / 'alright')
    d_th_idx = np.linalg.norm(pts[4] - pts[5])
    d_th_pky = np.linalg.norm(pts[4] - pts[17])
    d_base_pky = np.linalg.norm(pts[2] - pts[17])
    thumb_up = bool(pts[4, 1] < pts[2, 1] - span * 0.38 and d_th_idx > span * 0.35)
    thumb_spread = bool(d_th_idx > span * 0.42 and d_th_pky > d_base_pky * 1.05)
    thumb_ext = bool(thumb_spread or thumb_up)

    # IMPORTANT: Sum as integers to avoid NumPy numpy.bool_ logical-OR behavior!
    ext_count = int(thumb_ext) + int(index_ext) + int(middle_ext) + int(ring_ext) + int(pinky_ext)

    return {
        "pts": pts,
        "wrist": wrist,
        "span": span,
        "thumb_ext": thumb_ext,
        "index_ext": index_ext,
        "middle_ext": middle_ext,
        "ring_ext": ring_ext,
        "pinky_ext": pinky_ext,
        "ext_count": ext_count,
        "tips": {
            "thumb": pts[4],
            "index": pts[8],
            "middle": pts[12],
            "ring": pts[16],
            "pinky": pts[20]
        }
    }


# Backward compatibility alias
extract_hand_features = get_hand_metrics


def classify_isl_geometric(results):
    """
    High-precision anatomical & geometric rule engine for Indian Sign Language (ISL)
    two-handed manual alphabet, vowels, and single/two-handed digits.
    Returns: (predicted_char, confidence) or (None, 0.0)
    """
    if not results:
        return None, 0.0

    lh_lms = None
    rh_lms = None

    if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks and len(results.left_hand_landmarks.landmark) >= 21:
        lh_lms = results.left_hand_landmarks.landmark
    if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks and len(results.right_hand_landmarks.landmark) >= 21:
        rh_lms = results.right_hand_landmarks.landmark

    # Fallback for Hands solution
    if not lh_lms and not rh_lms and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            if len(hand_lms.landmark) >= 21:
                label = "Right"
                if hasattr(results, 'multi_handedness') and idx < len(results.multi_handedness):
                    label = results.multi_handedness[idx].classification[0].label
                if label == "Left" and not lh_lms:
                    lh_lms = hand_lms.landmark
                elif not rh_lms:
                    rh_lms = hand_lms.landmark

    lh = get_hand_metrics(lh_lms)
    rh = get_hand_metrics(rh_lms)

    if not lh and not rh:
        return None, 0.0

    # Check if both arms/wrists are raised in signing posture via pose landmarks
    both_wrists_raised = False
    if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
        pose_lms = results.pose_landmarks.landmark
        if len(pose_lms) > 16:
            l_wrist = pose_lms[15]
            r_wrist = pose_lms[16]
            l_vis = getattr(l_wrist, 'visibility', 1.0)
            r_vis = getattr(r_wrist, 'visibility', 1.0)
            # Both wrists are visible and raised into signing space (y < 0.88, not hanging down)
            if l_vis > 0.30 and r_vis > 0.30 and l_wrist.y < 0.88 and r_wrist.y < 0.88:
                if abs(l_wrist.y - r_wrist.y) < 0.35:
                    both_wrists_raised = True

    # -------------------------------------------------------------
    # A. TWO-HANDED ISL RECOGNITION (Vowels, Consonants & Digits)
    # -------------------------------------------------------------
    if lh and rh:
        l_span = lh["span"]
        r_span = rh["span"]
        avg_span = (l_span + r_span) / 2.0

        # Symmetrically evaluate both (rh=pointer, lh=base) and (lh=pointer, rh=base)
        # to ensure full robustness against handedness and MediaPipe left/right hand label flips
        pairs = [(rh, lh), (lh, rh)]

        # 1. Two-Handed ISL VOWELS (Pointer index touches base fingertips or pads of OPEN base hand)
        for pointer, base in pairs:
            # Pointer must strictly be an index pointer (middle, ring, pinky curled)
            # Base hand must be open (at least 3 fingers extended, distinguishing it from Letter D loop)
            if pointer["index_ext"] and not pointer["middle_ext"] and not pointer["pinky_ext"]:
                if base["ext_count"] >= 3 or (base["middle_ext"] and base["pinky_ext"]):
                    p_index = pointer["tips"]["index"]

                    # Check distance to fingertip AND distal joint/pad for each vowel.
                    d_A = min(np.linalg.norm(p_index - base["tips"]["thumb"]),
                              np.linalg.norm(p_index - base["pts"][3])) / avg_span
                    d_E = min(np.linalg.norm(p_index - base["tips"]["index"]),
                              np.linalg.norm(p_index - base["pts"][7])) / avg_span
                    d_I = min(np.linalg.norm(p_index - base["tips"]["middle"]),
                              np.linalg.norm(p_index - base["pts"][11])) / avg_span
                    d_O = min(np.linalg.norm(p_index - base["tips"]["ring"]),
                              np.linalg.norm(p_index - base["pts"][15])) / avg_span
                    d_U = min(np.linalg.norm(p_index - base["tips"]["pinky"]),
                              np.linalg.norm(p_index - base["pts"][19])) / avg_span

                    touch_thresh = 0.58
                    min_dist = min(d_A, d_E, d_I, d_O, d_U)
                    if min_dist < touch_thresh:
                        if min_dist == d_A:
                            return 'A', 0.95
                        elif min_dist == d_E:
                            return 'E', 0.95
                        elif min_dist == d_I:
                            return 'I', 0.95
                        elif min_dist == d_O:
                            return 'O', 0.95
                        elif min_dist == d_U:
                            return 'U', 0.95

        # 2. Letter B: Both hands index & thumb touch forming double circle
        d_b1 = np.linalg.norm(rh["tips"]["index"] - lh["tips"]["thumb"]) / avg_span
        d_b2 = np.linalg.norm(rh["tips"]["thumb"] - lh["tips"]["index"]) / avg_span
        d_ii = np.linalg.norm(rh["tips"]["index"] - lh["tips"]["index"]) / avg_span
        d_tt = np.linalg.norm(rh["tips"]["thumb"] - lh["tips"]["thumb"]) / avg_span
        if ((d_b1 < 0.52 and d_b2 < 0.52) or (d_ii < 0.45 and d_tt < 0.45)) and not rh["middle_ext"] and not lh["middle_ext"]:
            return 'B', 0.92

        # 3. Two-handed Consonants against base palm
        for p, b in pairs:
            p_palm = np.linalg.norm(p["tips"]["index"] - b["pts"][9]) / avg_span

            # Letter C: Curved hand against open palm
            if p_palm < 0.68 and not p["index_ext"] and not p["pinky_ext"]:
                return 'C', 0.88

            # Letter D: Vertical index against thumb-index loop
            if p["index_ext"] and not p["middle_ext"]:
                d_base_loop = np.linalg.norm(b["tips"]["thumb"] - b["tips"]["index"]) / avg_span
                d_touch_loop = min(np.linalg.norm(p["tips"]["index"] - b["tips"]["index"]),
                                   np.linalg.norm(p["tips"]["index"] - b["tips"]["thumb"]),
                                   np.linalg.norm(p["pts"][7] - b["tips"]["index"])) / avg_span
                if d_base_loop < 0.55 and d_touch_loop < 0.55:
                    return 'D', 0.91

            # Letter L: 'L' shape placed on palm
            if p["thumb_ext"] and p["index_ext"] and not p["middle_ext"] and not p["pinky_ext"]:
                if p_palm < 0.75 or np.linalg.norm(p["wrist"] - b["pts"][9]) / avg_span < 0.80:
                    return 'L', 0.93

            # Letter M & W: 3 fingers resting on palm (distinguished by spread)
            if p_palm < 0.72 and p["index_ext"] and p["middle_ext"] and p["ring_ext"] and not p["pinky_ext"]:
                d_spread_3 = np.linalg.norm(p["tips"]["index"] - p["tips"]["ring"]) / p["span"]
                if d_spread_3 >= 0.55:
                    return 'W', 0.91
                else:
                    return 'M', 0.91

            # Letter N & V: 2 fingers resting on palm (distinguished by spread)
            if p_palm < 0.72 and p["index_ext"] and p["middle_ext"] and not p["ring_ext"] and not p["pinky_ext"]:
                d_spread_2 = np.linalg.norm(p["tips"]["index"] - p["tips"]["middle"]) / p["span"]
                if d_spread_2 >= 0.32:
                    return 'V', 0.91
                else:
                    return 'N', 0.91

            # Letter Y: Index placed in thumb-index webbing
            d_web = min(
                np.linalg.norm(p["tips"]["index"] - (b["pts"][2] + b["pts"][5]) / 2.0),
                np.linalg.norm(p["pts"][7] - (b["pts"][2] + b["pts"][5]) / 2.0)
            ) / avg_span
            if p["index_ext"] and not p["middle_ext"] and d_web < 0.52:
                return 'Y', 0.91

        # Letter X: Crossed index fingers
        if rh["index_ext"] and lh["index_ext"] and not rh["middle_ext"] and not lh["middle_ext"]:
            d_cross = min(
                np.linalg.norm(rh["tips"]["index"] - lh["tips"]["index"]),
                np.linalg.norm(rh["pts"][7] - lh["pts"][7]),
                np.linalg.norm(rh["tips"]["index"] - lh["pts"][7]),
                np.linalg.norm(rh["pts"][7] - lh["tips"]["index"])
            ) / avg_span
            if d_cross < 0.48:
                return 'X', 0.92

        # SPACE: Both hands open flat with all 5 fingers extended facing camera
        if lh["ext_count"] == 5 and rh["ext_count"] == 5:
            wrist_dist = np.linalg.norm(lh["wrist"] - rh["wrist"]) / avg_span
            if wrist_dist > 1.2:
                return 'SPACE', 0.92

        # CRITICAL: If BOTH hands are active in view, DO NOT fall through to single-handed digits!
        # The user is gesturing with two hands; never misidentify pointer hand as single-handed '1'.
        return None, 0.0

    # -------------------------------------------------------------
    # B. SINGLE-HANDED ISL & MANUAL DIGITS (Dominant Right/Left Hand ONLY)
    # -------------------------------------------------------------
    # SAFETY GUARD: If pose landmarks show BOTH wrists raised in signing space,
    # the user is performing a two-handed sign where one hand was momentarily occluded/dropped.
    # NEVER fall through to single-handed digits (which would misclassify open base hand as '5').
    if both_wrists_raised:
        return None, 0.0

    active_hand = rh if rh is not None else lh
    if active_hand:
        ext = active_hand["ext_count"]
        # Digits 1 to 5 (ONLY when one hand is in view and other arm is down)
        if active_hand["index_ext"] and not active_hand["middle_ext"] and not active_hand["ring_ext"] and not active_hand["pinky_ext"] and not active_hand["thumb_ext"]:
            return '1', 0.92
        elif active_hand["index_ext"] and active_hand["middle_ext"] and not active_hand["ring_ext"] and not active_hand["pinky_ext"] and not active_hand["thumb_ext"]:
            return '2', 0.92
        elif active_hand["thumb_ext"] and active_hand["index_ext"] and active_hand["middle_ext"] and not active_hand["ring_ext"] and not active_hand["pinky_ext"]:
            return '3', 0.90
        elif active_hand["index_ext"] and active_hand["middle_ext"] and active_hand["ring_ext"] and active_hand["pinky_ext"] and not active_hand["thumb_ext"]:
            return '4', 0.92
        elif ext == 5:
            return '5', 0.88
        elif ext == 0:
            return '0', 0.85
        elif active_hand["thumb_ext"] and active_hand["index_ext"] and not active_hand["middle_ext"] and not active_hand["ring_ext"] and not active_hand["pinky_ext"]:
            return 'L', 0.89

    return None, 0.0


class ISLFingerspellingEngine:
    """
    Real-time Two-Handed ISL Fingerspelling Recognizer, Word Buffer Assembler,
    and Smart Dictionary Autocompletion / Autocorrection Engine.
    """
    def __init__(self):
        self.model = None
        self.classes = [c for c in ISL_ALPHABET if c not in ['SPACE', 'BACKSPACE']]
        self.current_word = ""
        self.history = []
        self.last_char = ""
        self.char_hold_count = 0
        self.hold_threshold = 5  # Consecutive frames to register a letter (~0.16s at 30fps)
        self.two_hand_cooldown = 0  # Prevents momentary single-hand occlusion from misclassifying as digit '5'
        self.dictionary = COMMON_ISL_DICTIONARY

    def predict(self, results, min_confidence=0.60):
        """
        Fast hybrid classifier: Combines geometric rule engine with trained features.
        """
        # 1. High precision geometric classifier
        geom_char, geom_conf = classify_isl_geometric(results)
        if geom_char and geom_conf >= min_confidence:
            return geom_char, geom_conf

        return None, 0.0

    def get_word_suggestions(self, max_suggestions=3):
        """
        Return top word completions or corrections based on the currently spelled prefix.
        """
        prefix = self.current_word.strip().upper()
        if not prefix:
            return []

        # 1. Exact prefix matches
        prefix_matches = [w for w in self.dictionary if w.startswith(prefix)]
        if prefix_matches:
            # Sort by length similarity
            prefix_matches.sort(key=lambda x: (len(x), x))
            return prefix_matches[:max_suggestions]

        # 2. Fuzzy edit distance matches (Levenshtein-like)
        def simple_dist(s1, s2):
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            distances = range(len(s1) + 1)
            for i2, c2 in enumerate(s2):
                distances_ = [i2+1]
                for i1, c1 in enumerate(s1):
                    if c1 == c2:
                        distances_.append(distances[i1])
                    else:
                        distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
                distances = distances_
            return distances[-1]

        fuzzy_matches = []
        for word in self.dictionary:
            dist = simple_dist(prefix, word[:len(prefix)])
            if dist <= 2:
                fuzzy_matches.append((dist, word))

        fuzzy_matches.sort(key=lambda x: (x[0], len(x[1])))
        return [w for _, w in fuzzy_matches[:max_suggestions]]

    def autocomplete(self, index=0):
        """Auto-complete current word buffer with top suggestion."""
        suggestions = self.get_word_suggestions()
        if suggestions and index < len(suggestions):
            self.current_word = suggestions[index] + " "
            return suggestions[index]
        return self.current_word

    def update_stream(self, results, min_confidence=0.60):
        """
        Process stream frame, stabilize predictions, and assemble spelled word.
        Returns: (predicted_char, current_assembled_word, is_new_letter)
        """
        # Detect whether two hands or both wrists are raised in front of torso
        has_two_hands = False
        if results:
            if hasattr(results, 'left_hand_landmarks') and hasattr(results, 'right_hand_landmarks'):
                if results.left_hand_landmarks and results.right_hand_landmarks:
                    has_two_hands = True
            if not has_two_hands and hasattr(results, 'multi_hand_landmarks') and results.multi_hand_landmarks:
                if len(results.multi_hand_landmarks) >= 2:
                    has_two_hands = True
            if not has_two_hands and hasattr(results, 'pose_landmarks') and results.pose_landmarks:
                pose_lms = results.pose_landmarks.landmark
                if len(pose_lms) > 16:
                    l_wrist = pose_lms[15]
                    r_wrist = pose_lms[16]
                    l_vis = getattr(l_wrist, 'visibility', 1.0)
                    r_vis = getattr(r_wrist, 'visibility', 1.0)
                    if l_vis > 0.30 and r_vis > 0.30 and l_wrist.y < 0.88 and r_wrist.y < 0.88:
                        if abs(l_wrist.y - r_wrist.y) < 0.35:
                            has_two_hands = True

        if has_two_hands:
            self.two_hand_cooldown = 18  # Hold guard for ~0.6 seconds

        char, conf = self.predict(results, min_confidence=min_confidence)

        # If in two-handed gesture window, suppress single-handed digits ('0'-'5')
        # caused by brief hand occlusion
        if self.two_hand_cooldown > 0:
            self.two_hand_cooldown -= 1
            if char in ['0', '1', '2', '3', '4', '5']:
                char = None

        is_new_letter = False

        if char:
            if char == self.last_char:
                self.char_hold_count += 1
            else:
                self.last_char = char
                self.char_hold_count = 1

            if self.char_hold_count == self.hold_threshold:
                # Registered letter
                if char == 'SPACE':
                    if self.current_word and not self.current_word.endswith(" "):
                        # Try autocorrect on space
                        suggestions = self.get_word_suggestions()
                        if suggestions and len(self.current_word.strip()) >= 3:
                            # Snap to top suggestion if exact word prefix
                            if suggestions[0].startswith(self.current_word.strip().upper()):
                                self.current_word = suggestions[0]
                        self.current_word += " "
                elif char == 'BACKSPACE':
                    self.current_word = self.current_word[:-1]
                else:
                    self.current_word += char
                is_new_letter = True
        else:
            self.char_hold_count = max(0, self.char_hold_count - 1)

        return char, self.current_word, is_new_letter

    def clear(self):
        self.current_word = ""
        self.last_char = ""
        self.char_hold_count = 0

    def backspace(self):
        self.current_word = self.current_word[:-1]
