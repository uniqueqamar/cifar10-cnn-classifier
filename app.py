"""
Common Object Image Classifier
--------------------------------
A Streamlit front-end for a CNN trained from scratch (TensorFlow/Keras) on
CIFAR-10, classifying everyday objects into 10 categories.

Run:
    streamlit run app.py

Expects a trained model at: outputs/model.keras
Optionally reads: outputs/training_curves.png, outputs/confusion_matrix.png,
                   outputs/classification_report.txt
"""

import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# ============================================================================
# CONFIG
# ============================================================================

MODEL_PATH = "outputs/model.keras"
CURVES_PATH = "outputs/training_curves.png"
CONFUSION_PATH = "outputs/confusion_matrix.png"
REPORT_PATH = "outputs/classification_report.txt"

CLASS_INFO = [
    ("Airplane", "✈️"),
    ("Automobile", "🚗"),
    ("Bird", "🐦"),
    ("Cat", "🐱"),
    ("Deer", "🦌"),
    ("Dog", "🐶"),
    ("Frog", "🐸"),
    ("Horse", "🐴"),
    ("Ship", "🚢"),
    ("Truck", "🚚"),
]
CLASS_NAMES = [name for name, _ in CLASS_INFO]
CLASS_EMOJIS = {name: emoji for name, emoji in CLASS_INFO}

st.set_page_config(
    page_title="Common Object Image Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# STYLING
# ============================================================================

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #0F172A;
            --bg-secondary: #1E293B;
            --accent-primary: #8B5CF6;
            --accent-secondary: #06B6D4;
            --text-light: #F1F5F9;
            --text-muted: #94A3B8;
        }

        .stApp {
            background: linear-gradient(160deg, #0F172A 0%, #0B1120 100%);
            color: var(--text-light);
        }

        /* Hide default Streamlit chrome for a cleaner product feel */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* ---------- Typography ---------- */
        h1, h2, h3, h4 {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            color: var(--text-light) !important;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #8B5CF6, #06B6D4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.25rem;
        }

        .hero-tagline {
            font-size: 1.15rem;
            color: var(--text-muted);
            margin-bottom: 2rem;
        }

        /* ---------- Glass cards ---------- */
        .glass-card {
            background: rgba(30, 41, 59, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: transform 0.2s ease, border 0.2s ease;
        }

        .glass-card:hover {
            transform: translateY(-3px);
            border: 1px solid rgba(139, 92, 246, 0.5);
        }

        .stat-card {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 1.2rem;
            text-align: center;
        }

        .stat-number {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #8B5CF6, #06B6D4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .stat-label {
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }

        /* ---------- Prediction card ---------- */
        .prediction-card {
            background: linear-gradient(135deg, rgba(139,92,246,0.18), rgba(6,182,212,0.12));
            border: 1px solid rgba(139, 92, 246, 0.4);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }

        .prediction-label {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--text-light);
            margin: 0.25rem 0;
        }

        .prediction-confidence {
            font-size: 1.1rem;
            color: var(--accent-secondary);
            font-weight: 600;
        }

        /* ---------- Confidence bars ---------- */
        .bar-row {
            display: flex;
            align-items: center;
            margin-bottom: 0.6rem;
            gap: 0.75rem;
        }

        .bar-label {
            width: 140px;
            font-size: 0.9rem;
            color: var(--text-light);
        }

        .bar-track {
            flex: 1;
            background: rgba(148, 163, 184, 0.15);
            border-radius: 8px;
            height: 18px;
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: 8px;
            background: linear-gradient(90deg, #8B5CF6, #06B6D4);
        }

        .bar-value {
            width: 55px;
            text-align: right;
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        /* ---------- Class chips ---------- */
        .class-chip {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            transition: transform 0.2s ease, border 0.2s ease;
        }

        .class-chip:hover {
            transform: translateY(-2px);
            border: 1px solid rgba(6, 182, 212, 0.5);
        }

        .class-chip-emoji {
            font-size: 1.8rem;
        }

        .class-chip-name {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: #0B1120;
            border-right: 1px solid rgba(148, 163, 184, 0.1);
        }

        /* ---------- Buttons ---------- */
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(90deg, #8B5CF6, #06B6D4);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.4rem;
            font-weight: 600;
            transition: opacity 0.2s ease;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            opacity: 0.85;
            color: white;
        }

        /* ---------- Footer ---------- */
        .app-footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            padding: 2rem 0 1rem 0;
            border-top: 1px solid rgba(148, 163, 184, 0.1);
            margin-top: 2rem;
        }

        .section-caption {
            color: var(--text-muted);
            font-size: 0.9rem;
            text-align: center;
            margin-top: -0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MODEL LOADING / INFERENCE
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model_cached():
    """Load the trained Keras model once and cache it across reruns."""
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception:
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to 32x32, normalize to [0,1], and add a batch dimension."""
    image = image.convert("RGB")
    image = image.resize((32, 32))
    array = np.array(image).astype("float32") / 255.0
    array = np.expand_dims(array, axis=0)
    return array


def predict_image(model, image_array: np.ndarray):
    """Run inference and return (predicted_class, confidence, all_probabilities)."""
    predictions = model.predict(image_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx])
    return predicted_class, confidence, predictions


def render_confidence_bars(class_names, probabilities) -> str:
    """Build HTML for horizontal confidence bars, sorted descending."""
    pairs = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)
    html = ""
    for name, prob in pairs:
        pct = prob * 100
        emoji = CLASS_EMOJIS.get(name, "")
        html += f"""
        <div class="bar-row">
            <div class="bar-label">{emoji} {name}</div>
            <div class="bar-track">
                <div class="bar-fill" style="width:{pct:.1f}%;"></div>
            </div>
            <div class="bar-value">{pct:.1f}%</div>
        </div>
        """
    return html


# ============================================================================
# SESSION STATE
# ============================================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================================
# SIDEBAR
# ============================================================================

def read_test_accuracy():
    """Parse the saved classification report for the test accuracy, if present."""
    if not os.path.exists(REPORT_PATH):
        return None
    try:
        with open(REPORT_PATH, "r") as f:
            first_line = f.readline()
        if "Test Accuracy" in first_line:
            value = first_line.split(":")[1].strip().replace("%", "")
            return float(value)
    except Exception:
        return None
    return None


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 Object Classifier")
        st.markdown(
            "A CNN trained from scratch on **CIFAR-10** to recognize "
            "10 everyday object categories."
        )

        st.markdown("---")
        st.markdown("### 📦 Model Info")
        model_status = "✅ Loaded" if load_model_cached() is not None else "⚠️ Not found"
        st.markdown(
            f"""
            <div class="glass-card">
                <b>Status:</b> {model_status}<br>
                <b>Framework:</b> TensorFlow / Keras<br>
                <b>Input size:</b> 32x32x3<br>
                <b>Classes:</b> 10<br>
                <b>Path:</b> <code>{MODEL_PATH}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 📊 Performance")
        test_acc = read_test_accuracy()
        acc_display = f"{test_acc:.2f}%" if test_acc is not None else "Run training to view"
        st.markdown(
            f"""
            <div class="glass-card">
                <b>Test Accuracy:</b> {acc_display}<br>
                <b>Optimizer:</b> Adam<br>
                <b>Loss:</b> Categorical Crossentropy
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["🏠 Home", "🔍 Classify", "📊 Analytics", "📁 Project Details", "📚 Class Reference"],
            label_visibility="collapsed",
        )

        if st.session_state.prediction_history:
            st.markdown("---")
            st.markdown(f"### 🕓 History ({len(st.session_state.prediction_history)})")
            history_df = pd.DataFrame(st.session_state.prediction_history)
            csv = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predictions (CSV)",
                data=csv,
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

        return page


# ============================================================================
# PAGE SECTIONS
# ============================================================================

def render_home():
    st.markdown('<div class="hero-title">Common Object Image Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-tagline">Deep Learning powered image recognition '
        'using a custom CNN trained on CIFAR-10</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    stats = [
        ("10", "Classes"),
        ("60,000", "Images"),
        ("CNN", "From Scratch"),
        ("TF / Keras", "Framework"),
    ]
    for col, (number, label) in zip(cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-number">{number}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 🧬 About the Model")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h4>📁 Dataset</h4>
                <p style="color:#94A3B8;">
                CIFAR-10 — 60,000 32x32 color images across 10 balanced classes
                (50,000 train / 10,000 test). Split further here into
                45,000 train / 5,000 validation / 10,000 test, with the test
                set touched only once for final evaluation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h4>🏗️ Architecture</h4>
                <p style="color:#94A3B8; line-height:1.9;">
                Conv2D(32) → BatchNorm → MaxPooling<br>
                Conv2D(64) → BatchNorm → MaxPooling<br>
                Conv2D(128) → BatchNorm → MaxPooling<br>
                Flatten → Dense → Dropout → Softmax
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_classify():
    st.markdown("## 🔍 Try the Classifier")
    st.markdown(
        '<p class="section-caption">Upload a photo of a common object and the model will predict its class.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    model = load_model_cached()
    if model is None:
        st.error(
            f"⚠️ Model not found at `{MODEL_PATH}`. Train the model first "
            f"(see `train.py`) and make sure `outputs/model.keras` exists."
        )
        return

    uploaded_file = st.file_uploader(
        "Drag and drop an image, or click to browse",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("👆 Upload a JPG, JPEG, or PNG image to get started.")
        return

    try:
        image = Image.open(io.BytesIO(uploaded_file.read()))
    except Exception:
        st.error("⚠️ That file couldn't be read as an image. Please try a different file.")
        return

    col_img, col_pred = st.columns([1, 1.3])

    with col_img:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_pred:
        with st.spinner("Running inference..."):
            try:
                processed = preprocess_image(image)
                predicted_class, confidence, all_probs = predict_image(model, processed)
            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
                return

        st.success("✅ Prediction complete")

        emoji = CLASS_EMOJIS.get(predicted_class, "")
        st.markdown(
            f"""
            <div class="prediction-card">
                <div style="font-size:2.5rem;">{emoji}</div>
                <div class="prediction-label">{predicted_class}</div>
                <div class="prediction-confidence">{confidence*100:.1f}% confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Top 5 Predictions")
        top5_idx = np.argsort(all_probs)[::-1][:5]
        top5_names = [CLASS_NAMES[i] for i in top5_idx]
        top5_probs = [all_probs[i] for i in top5_idx]
        st.markdown(render_confidence_bars(top5_names, top5_probs), unsafe_allow_html=True)

        st.session_state.prediction_history.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": uploaded_file.name,
                "predicted_class": predicted_class,
                "confidence_pct": round(confidence * 100, 2),
            }
        )


def render_analytics():
    st.markdown("## 📊 Visual Analytics")
    st.markdown(
        '<p class="section-caption">Training behavior and error patterns from the saved run.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if os.path.exists(CURVES_PATH):
            st.image(CURVES_PATH, caption="Training & validation accuracy / loss", use_container_width=True)
        else:
            st.warning(f"`{CURVES_PATH}` not found. Run `train.py` to generate it.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if os.path.exists(CONFUSION_PATH):
            st.image(CONFUSION_PATH, caption="Confusion matrix (test set)", use_container_width=True)
        else:
            st.warning(f"`{CONFUSION_PATH}` not found. Run `train.py` to generate it.")
        st.markdown("</div>", unsafe_allow_html=True)

    if os.path.exists(REPORT_PATH):
        with st.expander("📄 Full classification report"):
            with open(REPORT_PATH, "r") as f:
                st.text(f.read())


def render_project_details():
    st.markdown("## 📁 Project Details")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h4>📦 Dataset</h4>
                <p style="color:#94A3B8;">CIFAR-10<br>10 balanced classes<br>32x32 RGB images</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h4>⚙️ Training</h4>
                <p style="color:#94A3B8;">
                Adam Optimizer<br>
                Categorical Crossentropy<br>
                EarlyStopping<br>
                ReduceLROnPlateau<br>
                ModelCheckpoint
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="glass-card">
                <h4>🔄 Augmentation</h4>
                <p style="color:#94A3B8;">
                Rotation<br>
                Horizontal Flip<br>
                Zoom<br>
                Width Shift<br>
                Height Shift
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_class_reference():
    st.markdown("## 📚 Class Reference")
    st.markdown(
        '<p class="section-caption">All 10 categories the model can recognize.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(5)
    for i, (name, emoji) in enumerate(CLASS_INFO):
        with cols[i % 5]:
            st.markdown(
                f"""
                <div class="class-chip">
                    <div class="class-chip-emoji">{emoji}</div>
                    <div class="class-chip-name">{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)


def render_footer():
    st.markdown(
        '<div class="app-footer">Built with TensorFlow, Keras, and Streamlit</div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():
    inject_custom_css()

    if not TF_AVAILABLE:
        st.error(
            "⚠️ TensorFlow is not installed in this environment. "
            "Install it with `pip install tensorflow` to enable predictions."
        )

    page = render_sidebar()

    if page == "🏠 Home":
        render_home()
    elif page == "🔍 Classify":
        render_classify()
    elif page == "📊 Analytics":
        render_analytics()
    elif page == "📁 Project Details":
        render_project_details()
    elif page == "📚 Class Reference":
        render_class_reference()

    render_footer()


if __name__ == "__main__":
    main()