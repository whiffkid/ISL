import os
import sys
import glob

# Ensure matching local venv site-packages are accessible
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
for venv_site in glob.glob(os.path.join(base_dir, "venv*", "lib*", py_ver, "site-packages")):
    if venv_site not in sys.path and os.path.exists(venv_site):
        sys.path.insert(0, venv_site)

import time
import json
import numpy as np
import cv2
import streamlit as st
import tensorflow as tf

from utils import (
    mediapipe_detection,
    draw_styled_landmarks,
    landmarks_data,
    is_hand_active,
    prob_viz,
    TranscriptRecorder,
    is_emergency_sign
)
from models import load_model, get_actions
from fingerspelling import ISLFingerspellingEngine, ISL_ALPHABET
from grammar_translator import ISLGrammarTranslator
from isl_learning_guide import (
    ISL_SYMBOL_GUIDES,
    get_symbol_categories,
    generate_hand_svg,
    evaluate_hand_posture,
    draw_tutor_camera_overlay
)
import mediapipe as mp

st.set_page_config(
    page_title="Universal Real-Time ISL Translation Studio",
    page_icon=":material/sign_language:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Polished Modern Dark Theme Styles
st.markdown("""
<style>
    /* Hero Header */
    .hero-container {
        padding: 10px 0 18px 0;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(120deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #94a3b8;
        font-weight: 400;
        margin: 0 0 12px 0;
    }
    
    /* Translation Card */
    .trans-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
    }
    .trans-lang-tag {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #38bdf8;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .trans-output {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.35;
        min-height: 48px;
    }
    .gloss-chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 12px;
        align-items: center;
    }
    .gloss-chip {
        background: #0f172a;
        color: #93c5fd;
        border: 1px solid #1e3a8a;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.88rem;
        font-weight: 600;
    }
    .spelled-chip {
        background: #14532d;
        color: #86efac;
        border: 1px solid #16a34a;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.88rem;
        font-weight: 600;
    }
    
    /* Emergency Alert */
    .emergency-alert {
        background: linear-gradient(135deg, #991b1b 0%, #7f1d1d 100%);
        border: 2px solid #ef4444;
        padding: 14px 18px;
        border-radius: 10px;
        color: #fef2f2;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 16px;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.4);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Coach Feedback Card */
    .coach-box {
        padding: 16px 20px;
        border-radius: 10px;
        font-size: 1.08rem;
        font-weight: 600;
        margin-bottom: 14px;
        border-left: 5px solid;
    }
    .coach-perfect {
        background: #064e3b;
        color: #a7f3d0;
        border-color: #10b981;
    }
    .coach-good {
        background: #451a03;
        color: #fde68a;
        border-color: #f59e0b;
    }
    .coach-needs-work {
        background: #1e293b;
        color: #bfdbfe;
        border-color: #38bdf8;
    }
    
    /* Finger Anatomy Box */
    .anatomy-box {
        background: #090d16;
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #1e293b;
        font-size: 0.88rem;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    .anatomy-box b {
        color: #38bdf8;
    }
    
    /* SVG Visual Showcase Card */
    .svg-container {
        background: #090d16;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_sign_model(model_name="bilstm_attention"):
    actions = get_actions(model_name=model_name)
    try:
        model = load_model(model_name, pretrained=True, training=False, num_classes=len(actions))
    except Exception:
        model = load_model(model_name, pretrained=False, training=False, num_classes=len(actions))
    return model, actions


def render_header():
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Real-time ISL translation studio</h1>
        <p class="hero-subtitle">Multi-modal Indian Sign Language recognition, two-handed fingerspelling, and multilingual Indic translation</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    c1.badge("BiLSTM + Attention", icon=":material/psychology:", color="blue")
    c2.badge("Two-handed ISL", icon=":material/sign_language:", color="green")
    c3.badge("10 Indic languages", icon=":material/g_translate:", color="orange")
    c4.badge("AI posture coach", icon=":material/school:", color="purple")
    st.space(12)


def render_learn_tab(complexity=0, left_handed=False, enable_tts=True):
    categories = get_symbol_categories()
    cat_names = list(categories.keys())

    # Category Selection
    selected_cat = st.segmented_control(
        "Signal category",
        cat_names,
        default=cat_names[0],
        help="Select a category to view and practice specific hand signals."
    )
    if not selected_cat:
        selected_cat = cat_names[0]

    symbols_in_cat = categories[selected_cat]

    # Symbol Selection with Prev/Next controls
    col_prev, col_sym, col_next = st.columns([1, 6, 1])
    
    curr_idx = st.session_state.get("learn_sym_idx", 0)
    if curr_idx >= len(symbols_in_cat):
        curr_idx = 0
        st.session_state.learn_sym_idx = 0

    with col_prev:
        if st.button("Prev", icon=":material/arrow_back:", width="stretch", help="Previous hand signal"):
            curr_idx = (curr_idx - 1) % len(symbols_in_cat)
            st.session_state.learn_sym_idx = curr_idx
            st.rerun()

    with col_sym:
        selected_symbol = st.pills(
            "Select sign",
            symbols_in_cat,
            default=symbols_in_cat[curr_idx],
            label_visibility="collapsed"
        )
        if selected_symbol in symbols_in_cat:
            new_idx = symbols_in_cat.index(selected_symbol)
            if new_idx != curr_idx:
                st.session_state.learn_sym_idx = new_idx
                curr_idx = new_idx
        else:
            selected_symbol = symbols_in_cat[curr_idx]

    with col_next:
        if st.button("Next", icon=":material/arrow_forward:", width="stretch", help="Next hand signal"):
            curr_idx = (curr_idx + 1) % len(symbols_in_cat)
            st.session_state.learn_sym_idx = curr_idx
            st.rerun()

    guide = ISL_SYMBOL_GUIDES.get(selected_symbol, ISL_SYMBOL_GUIDES['A'])

    # Header Card with Summary
    with st.container(border=True):
        col_title, col_badges = st.columns([3, 1])
        with col_title:
            st.subheader(f"{guide['name']}")
            st.markdown(f"**Description:** {guide['summary']}")
            target_pt = guide.get('target_touch', 'N/A')
            st.caption(f":material/touch_app: Target touch: :orange[{target_pt}]")
        with col_badges:
            badge_text = "Two-handed sign" if guide.get('two_handed') else "Single-handed sign"
            badge_icon = ":material/sign_language:" if guide.get('two_handed') else ":material/pan_tool:"
            st.badge(badge_text, icon=badge_icon, color="blue")
            st.badge(guide.get('category', 'Alphabet'), icon=":material/category:", color="gray")

    col_guide, col_practice = st.columns([1, 1], gap="medium")

    with col_guide:
        with st.container(border=True):
            st.markdown("#### :material/architecture: Visual finger anatomy")
            
            # 1. High-Contrast SVG Visual Diagram
            svg_code = generate_hand_svg(selected_symbol)
            st.markdown(f"<div class='svg-container'>{svg_code}</div>", unsafe_allow_html=True)
            st.caption("Visual diagram illustrating target finger positions and contact point.")

            # 2. Step-by-Step Instructions
            with st.expander("Step-by-step instructions", expanded=True):
                for step in guide.get('steps', []):
                    st.markdown(f":material/check_small: {step}")

            # 3. Finger-by-Finger Anatomical Breakdown
            with st.expander("Finger-by-finger breakdown", expanded=True):
                if guide.get('two_handed'):
                    col_lh, col_rh = st.columns(2)
                    with col_lh:
                        st.markdown("**:material/pan_tool: Left hand (base):**")
                        lh_req = guide.get('left_hand_req', {})
                        for k, v in lh_req.items():
                            st.markdown(f"<div class='anatomy-box'><b>{k.capitalize()}:</b> {v}</div>", unsafe_allow_html=True)
                    with col_rh:
                        st.markdown("**:material/touch_app: Right hand (pointer):**")
                        rh_req = guide.get('right_hand_req', {})
                        for k, v in rh_req.items():
                            st.markdown(f"<div class='anatomy-box'><b>{k.capitalize()}:</b> {v}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**{guide['dominant_hand']} posture:**")
                    req = guide.get('right_hand_req', {})
                    cols = st.columns(2)
                    for idx, (k, v) in enumerate(req.items()):
                        cols[idx % 2].markdown(f"<div class='anatomy-box'><b>{k.capitalize()}:</b> {v}</div>", unsafe_allow_html=True)

            # 4. Pro Alignment Tips
            if guide.get('tips'):
                with st.expander("Pro alignment tips", expanded=False):
                    for tip in guide['tips']:
                        st.info(tip, icon=":material/lightbulb:")

    with col_practice:
        with st.container(border=True):
            st.markdown("#### :material/videocam: Live posture coach")
            st.caption("Turn on the webcam to receive real-time anatomical feedback and targeting crosshairs.")

            col_toggle, col_stars = st.columns([2, 1])
            with col_toggle:
                run_tutor_cam = st.toggle("Start posture coach camera", value=False, key=f"tutor_cam_{selected_symbol}")
            with col_stars:
                st.metric("Mastery stars", f"{st.session_state.get('tutor_stars', 0)} ⭐")

            score_progress = st.progress(0, text="Posture accuracy: 0%")
            hint_card = st.empty()
            checklist_placeholder = st.empty()
            tutor_frame_window = st.image([])

            if not run_tutor_cam:
                hint_card.markdown("""
                <div class="coach-box coach-needs-work">
                    :material/info: Toggle the camera switch above when you're ready to practice this hand signal.
                </div>
                """, unsafe_allow_html=True)

            if run_tutor_cam:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                mp_holistic = mp.solutions.holistic
                if "tutor_stars" not in st.session_state:
                    st.session_state.tutor_stars = 0
                if "high_score_streak" not in st.session_state:
                    st.session_state.high_score_streak = 0

                with mp_holistic.Holistic(
                    model_complexity=complexity,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    refine_face_landmarks=False
                ) as holistic:
                    while run_tutor_cam and cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            continue

                        image, results = mediapipe_detection(frame, holistic)
                        draw_styled_landmarks(image, results)

                        # Evaluate posture against chosen symbol
                        eval_res = evaluate_hand_posture(results, selected_symbol)
                        acc = eval_res['accuracy']
                        status = eval_res['status']
                        hint = eval_res['primary_hint']

                        # Render augmented camera HUD overlay with target crosshair & lines
                        draw_tutor_camera_overlay(image, eval_res, selected_symbol)

                        # Streak and Mastery tracking
                        if acc >= 80.0:
                            st.session_state.high_score_streak += 1
                            if st.session_state.high_score_streak == 25:
                                st.session_state.tutor_stars += 1
                                st.toast(f"🌟 Awesome! You mastered '{selected_symbol}'! Star awarded!")
                        else:
                            st.session_state.high_score_streak = 0

                        # Convert to RGB and display frame
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        tutor_frame_window.image(image_rgb, channels="RGB", width="stretch")

                        # Update Accuracy Progress Bar
                        score_progress.progress(int(acc), text=f"Posture accuracy: {int(acc)}% ({status})")

                        # Dynamic Coach Card styling
                        card_class = "coach-perfect" if status == "PERFECT" else ("coach-good" if status == "GOOD" else "coach-needs-work")
                        hint_card.markdown(f"""
                        <div class="coach-box {card_class}">
                            {hint}
                        </div>
                        """, unsafe_allow_html=True)

                        # Live Checklist
                        checklist_md = "<b>Live Anatomical Checklist:</b><br>"
                        for item in eval_res['checklist']:
                            icon = "✅" if item['passed'] else "❌"
                            checklist_md += f"{icon} <b>{item['label']}</b>: <span style='color:#94a3b8;'>{item['detail']}</span><br>"
                        checklist_placeholder.markdown(f"<div class='anatomy-box' style='padding:12px 16px;'>{checklist_md}</div>", unsafe_allow_html=True)

                cap.release()


def render_library_tab():
    st.subheader("Sign library & manual alphabet reference")
    st.caption("Browse all standard ISL two-handed vowels, consonants, numbers, and dynamic conversational signs.")

    lib_tabs = st.tabs([
        ":material/spellcheck: Vowels",
        ":material/font_download: Consonants",
        ":material/pin: Digits",
        ":material/chat: Common signs"
    ])

    categories = get_symbol_categories()

    with lib_tabs[0]:
        vowels = categories.get("Vowels (A, E, I, O, U)", ['A', 'E', 'I', 'O', 'U'])
        cols = st.columns(len(vowels))
        for idx, v in enumerate(vowels):
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"### `{v}`")
                    g = ISL_SYMBOL_GUIDES.get(v, {})
                    st.markdown(f"<div style='transform:scale(0.8); transform-origin:top left;'>{generate_hand_svg(v)}</div>", unsafe_allow_html=True)
                    st.caption(g.get('summary', ''))
                    st.badge(f"Touch: {g.get('target_touch', '')}", color="blue")

    with lib_tabs[1]:
        consonants = categories.get("Consonants", [])
        num_cols = 4
        cols = st.columns(num_cols)
        for idx, c in enumerate(consonants):
            with cols[idx % num_cols]:
                with st.container(border=True):
                    st.markdown(f"#### `{c}`")
                    g = ISL_SYMBOL_GUIDES.get(c, {})
                    st.caption(g.get('summary', ''))
                    st.badge(f"Target: {g.get('target_touch', '')}", color="gray")

    with lib_tabs[2]:
        digits = categories.get("Numbers (0–5)", ['0', '1', '2', '3', '4', '5'])
        cols = st.columns(len(digits))
        for idx, d in enumerate(digits):
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"### `{d}`")
                    g = ISL_SYMBOL_GUIDES.get(d, {})
                    st.caption(g.get('summary', ''))
                    st.badge("Single-handed", color="purple")

    with lib_tabs[3]:
        common = categories.get("Common Signs", ['hello', 'thank you', 'alright'])
        cols = st.columns(len(common))
        for idx, cs in enumerate(common):
            with cols[idx]:
                with st.container(border=True):
                    st.markdown(f"#### {cs.upper()}")
                    g = ISL_SYMBOL_GUIDES.get(cs, {})
                    st.caption(g.get('summary', ''))
                    st.badge("Dynamic sign", color="green")


def main():
    render_header()

    # Sidebar Controls
    with st.sidebar:
        st.markdown("### :material/settings: Settings")

        with st.container(border=True):
            st.markdown("**:material/memory: AI model**")
            model_choice = st.selectbox(
                "Sign sequence model",
                ["bilstm_attention", "lstm_v3", "lstm_v1", "transformer"],
                index=0,
                label_visibility="collapsed"
            )
            complexity = st.select_slider(
                "Model complexity (0 = fastest)",
                [0, 1, 2],
                value=0,
                help="Higher values increase landmark precision at the cost of frame rate."
            )
            confidence_thresh = st.slider(
                "Prediction threshold",
                0.30, 0.95, 0.55, 0.05,
                help="Minimum model confidence needed to commit a dynamic sign."
            )

        with st.container(border=True):
            st.markdown("**:material/accessibility: Accessibility**")
            left_handed = st.checkbox("Left-handed signer mode", value=False)
            enable_tts = st.checkbox("Voice speech synthesis (TTS)", value=True)

        with st.container(border=True):
            st.markdown("**:material/info: System status**")
            st.badge("MediaPipe Holistic active", icon=":material/check_circle:", color="green")
            st.badge("100% On-device private", icon=":material/security:", color="blue")
            st.caption("No video frames leave your browser.")

    # Load Model and Engines
    model, actions = load_sign_model(model_choice)
    if "spell_engine" not in st.session_state:
        st.session_state.spell_engine = ISLFingerspellingEngine()
    spell_engine = st.session_state.spell_engine

    if "grammar_engine" not in st.session_state:
        st.session_state.grammar_engine = ISLGrammarTranslator()
    grammar_engine = st.session_state.grammar_engine

    if "session_words" not in st.session_state:
        st.session_state.session_words = []
    if "transcript_recorder" not in st.session_state:
        st.session_state.transcript_recorder = TranscriptRecorder()
    if "last_sentence" not in st.session_state:
        st.session_state.last_sentence = ""

    # Supported Languages
    supported_langs = grammar_engine.get_supported_languages()
    lang_codes = list(supported_langs.keys())
    lang_labels = [f"{supported_langs[k]} ({k.upper()})" for k in lang_codes]

    # Main Navigation Tabs
    tab_translate, tab_learn, tab_library = st.tabs([
        ":material/translate: Live translation studio",
        ":material/school: Learn & practice tutor",
        ":material/menu_book: Sign library & reference"
    ])

    with tab_translate:
        col_video, col_output = st.columns([1.15, 0.85], gap="large")

        # LEFT COLUMN: Video Stream & Live Controls
        with col_video:
            with st.container(border=True):
                col_feed_hdr, col_feed_badge = st.columns([2, 1])
                with col_feed_hdr:
                    st.markdown("#### :material/videocam: Camera feed")
                with col_feed_badge:
                    cam_state = "Active" if st.session_state.get("run_cam_state", False) else "Standby"
                    cam_badge_color = "green" if st.session_state.get("run_cam_state", False) else "gray"
                    st.badge(f"Feed {cam_state}", color=cam_badge_color)

                run_camera = st.toggle("Enable translation camera", value=False, key="run_cam_toggle")
                st.session_state.run_cam_state = run_camera

                frame_window = st.image([])

                # Recognition Mode Switcher & Quick Actions Bar
                col_mode_label, col_mode_ctrl = st.columns([1, 2])
                with col_mode_label:
                    st.markdown("**:material/tune: Mode:**")
                with col_mode_ctrl:
                    mode_choice = st.segmented_control(
                        "Recognition mode",
                        ["Hybrid", "Signs only", "Fingerspelling"],
                        default="Hybrid",
                        label_visibility="collapsed"
                    )

                # Quick Actions Bar
                act_col1, act_col2, act_col3 = st.columns(3)
                if act_col1.button("Backspace", icon=":material/backspace:", width="stretch"):
                    spell_engine.backspace()
                    st.rerun()
                if act_col2.button("Add space", icon=":material/space_bar:", width="stretch"):
                    if spell_engine.current_word and not spell_engine.current_word.endswith(" "):
                        spell_engine.current_word += " "
                    st.rerun()
                if act_col3.button("Clear session", icon=":material/delete:", width="stretch"):
                    st.session_state.session_words = []
                    spell_engine.clear()
                    st.session_state.last_sentence = ""
                    st.rerun()

        # RIGHT COLUMN: Translation & Speech Center
        with col_output:
            with st.container(border=True):
                st.markdown("#### :material/g_translate: Translation output")
                
                selected_lang_label = st.selectbox(
                    "Target output language",
                    lang_labels,
                    index=0
                )
                target_lang = lang_codes[lang_labels.index(selected_lang_label)]
                grammar_engine.set_target_language(target_lang)

                emergency_placeholder = st.empty()
                translation_placeholder = st.empty()
                gloss_placeholder = st.empty()
                suggestions_placeholder = st.empty()

            # Transcript & Session Record Container
            with st.container(border=True):
                col_rec_hdr, col_rec_cnt = st.columns([2, 1])
                with col_rec_hdr:
                    st.markdown("#### :material/history: Session transcript")
                with col_rec_cnt:
                    rec_count = len(st.session_state.transcript_recorder.records)
                    st.badge(f"{rec_count} entries", color="blue")

                if st.session_state.transcript_recorder.records:
                    rec_list = list(reversed(st.session_state.transcript_recorder.records[-5:]))
                    for st_t, et_t, glosses, text in rec_list:
                        st.markdown(f"**`{int(st_t//60):02d}:{int(st_t%60):02d}`** {text} *(ISL: {', '.join(glosses)})*")

                    dl_col1, dl_col2 = st.columns(2)
                    srt_data = ""
                    for i, (st_t, et_t, glosses, text) in enumerate(st.session_state.transcript_recorder.records, 1):
                        srt_data += f"{i}\n{st.session_state.transcript_recorder._format_srt_time(st_t)} --> {st.session_state.transcript_recorder._format_srt_time(et_t)}\n{text}\n\n"

                    dl_col1.download_button(
                        label="Download .srt",
                        icon=":material/subtitles:",
                        data=srt_data,
                        file_name="isl_subtitles.srt",
                        mime="text/plain",
                        width="stretch"
                    )

                    txt_data = f"=== ISL Translation Session Transcript ===\n\n"
                    for st_t, et_t, glosses, text in st.session_state.transcript_recorder.records:
                        txt_data += f"[{int(st_t // 60):02d}:{int(st_t % 60):02d}] ISL: [{glosses}] -> {text}\n"

                    dl_col2.download_button(
                        label="Download .txt",
                        icon=":material/description:",
                        data=txt_data,
                        file_name="isl_transcript.txt",
                        mime="text/plain",
                        width="stretch"
                    )
                else:
                    st.caption("Recognized sentences will automatically be recorded here.")

        # Real-Time Processing Loop
        if run_camera:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            mp_holistic = mp.solutions.holistic
            sequence = []
            frame_idx = 0

            @tf.function(reduce_retracing=True)
            def fast_web_predict(x):
                return model(x, training=False)

            _ = fast_web_predict(np.zeros((1, 30, 150), dtype=np.float32))

            mode_key = "hybrid" if mode_choice == "Hybrid" else ("sign" if mode_choice == "Sign sequences" else "spell")

            with mp_holistic.Holistic(
                model_complexity=complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_face_landmarks=False
            ) as holistic:
                last_web_gloss_key = None
                translated_text = ""
                suggestions = []
                web_cooldown = 0
                web_rolling_preds = []
                while run_camera and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue

                    frame_idx += 1
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    hand_active = is_hand_active(results)
                    keypoints = landmarks_data(results, left_handed=left_handed)
                    sequence.append(keypoints)
                    sequence = sequence[-60:]

                    # Dynamic Sign Recognition
                    if mode_key in ["sign", "hybrid"]:
                        web_cooldown = max(0, web_cooldown - 1)
                        if len(sequence) == 60 and (frame_idx % 2 == 0) and hand_active and web_cooldown == 0:
                            sampled_seq = np.array(sequence)[::2]
                            input_seq = np.expand_dims(sampled_seq, axis=0)
                            raw_preds = fast_web_predict(input_seq).numpy()[0]
                            web_rolling_preds.append(raw_preds)
                            if len(web_rolling_preds) > 3:
                                web_rolling_preds = web_rolling_preds[-3:]
                            preds = np.mean(web_rolling_preds, axis=0)

                            max_idx = int(np.argmax(preds))
                            conf = preds[max_idx]

                            if conf > confidence_thresh and max_idx < len(actions):
                                act = actions[max_idx]
                                if not st.session_state.session_words or act != st.session_state.session_words[-1]:
                                    st.session_state.session_words.append(act)
                                    if len(st.session_state.session_words) > 8:
                                        st.session_state.session_words = st.session_state.session_words[-8:]

                                sequence = sequence[-16:]
                                web_rolling_preds = []
                                web_cooldown = 18

                    # Fingerspelling Recognition
                    if mode_key in ["spell", "hybrid"] and hand_active:
                        char, curr_word, is_new = spell_engine.update_stream(results)
                        if is_new:
                            suggestions = spell_engine.get_word_suggestions(max_suggestions=3)
                    elif not spell_engine.current_word:
                        suggestions = []

                    # Multilingual Translation (only computed on state change)
                    curr_key = (tuple(st.session_state.session_words), spell_engine.current_word.strip(), target_lang)
                    if curr_key != last_web_gloss_key:
                        last_web_gloss_key = curr_key
                        full_gloss = list(st.session_state.session_words)
                        if spell_engine.current_word.strip():
                            full_gloss.append(spell_engine.current_word.strip())

                        translated_text, eng_text = grammar_engine.translate(full_gloss, target_lang=target_lang)
                        
                        if translated_text and translated_text != st.session_state.last_sentence:
                            st.session_state.last_sentence = translated_text
                            st.session_state.transcript_recorder.add_entry(full_gloss, translated_text)

                    # Check Emergency SOS
                    is_sos = any(is_emergency_sign(w) for w in st.session_state.session_words)

                    # Render Video Frame
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    frame_window.image(image_rgb, channels="RGB", width="stretch")

                    # Update Output Display Cards
                    if is_sos:
                        emergency_placeholder.markdown("""
                        <div class="emergency-alert">
                            :material/emergency: <b>EMERGENCY / DISTRESS SIGN DETECTED</b>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        emergency_placeholder.empty()

                    trans_display = translated_text if translated_text else "Waiting for signs..."
                    translation_placeholder.markdown(f"""
                    <div class="trans-card">
                        <div class="trans-lang-tag">{supported_langs.get(target_lang, 'TRANSLATION')}</div>
                        <div class="trans-output">{trans_display}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    gloss_chips = "".join([f"<span class='gloss-chip'>{w}</span>" for w in st.session_state.session_words])
                    spelled_chip = f"<span class='spelled-chip'>✍️ {spell_engine.current_word}</span>" if spell_engine.current_word else ""
                    gloss_placeholder.markdown(f"""
                    <div class="gloss-chip-container">
                        <span style="color:#94a3b8; font-size:0.85rem; font-weight:600;">Glosses:</span>
                        {gloss_chips if gloss_chips else "<span style='color:#64748b;'>None</span>"}
                        {spelled_chip}
                    </div>
                    """, unsafe_allow_html=True)

                    if suggestions:
                        sug_str = " ".join([f"<span class='spelled-chip' style='background:#1e293b; color:#38bdf8; border-color:#0284c7;'>{w}</span>" for w in suggestions])
                        suggestions_placeholder.markdown(f"<div style='margin-top:8px;'><span style='color:#94a3b8; font-size:0.85rem;'>Suggestions:</span> {sug_str}</div>", unsafe_allow_html=True)
                    else:
                        suggestions_placeholder.empty()

                cap.release()

    with tab_learn:
        render_learn_tab(complexity=complexity, left_handed=left_handed, enable_tts=enable_tts)

    with tab_library:
        render_library_tab()


if __name__ == "__main__":
    main()
