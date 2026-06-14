
import io
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

import pandas as pd
import streamlit as st

from disease_info import get_info, pretty_label

MODEL_PATH = os.path.join(os.path.dirname(__file__), "checkpoint_epoch4.pth")

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#f6fbf3 0%, #ffffff 60%); }
      .hero {
        background: linear-gradient(120deg,#2e7d32 0%, #66bb6a 100%);
        padding: 1.6rem 2rem; border-radius: 18px; color: white;
        box-shadow: 0 8px 24px rgba(46,125,50,0.25); margin-bottom: 1.4rem;
      }
      .hero h1 { color: white; margin: 0; font-size: 2.1rem; }
      .hero p  { color: #e8f5e9; margin: .4rem 0 0; font-size: 1.02rem; }
      .result-card {
        background: #ffffff; border: 1px solid #c8e6c9; border-left: 7px solid #2e7d32;
        border-radius: 14px; padding: 1.2rem 1.4rem; box-shadow: 0 4px 16px rgba(0,0,0,0.06);
      }
      .healthy { border-left-color:#2e7d32 !important; }
      .diseased { border-left-color:#e65100 !important; }
      .pill {
        display:inline-block; padding:.28rem .8rem; border-radius:999px;
        font-weight:600; font-size:.86rem; margin-bottom:.5rem;
      }
      .pill-ok  { background:#e8f5e9; color:#2e7d32; }
      .pill-bad { background:#fff3e0; color:#e65100; }
      .stProgress > div > div > div > div { background-color:#2e7d32; }
      .tip { background:#f1f8e9; border-radius:10px; padding:.8rem 1rem; margin-top:.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Model (cached so it loads once)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    class_names = ckpt["classes"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, class_names, device


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image: Image.Image, model, class_names, device, k=3):
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].cpu()
    top = torch.topk(probs, k)
    return [(class_names[i], float(probs[i])) for i in top.indices], probs
with st.sidebar:
    st.markdown("### 🌱 About")
    st.write(
        "This tool uses a **MobileNetV2** convolutional neural network "
        "fine-tuned on the **PlantVillage** dataset to identify leaf diseases "
        "of **pepper, potato and tomato**."
    )
    st.markdown("---")
    st.markdown("### 📋 How to use")
    st.write("1. Upload a clear photo of **one** leaf.\n"
             "2. Read the predicted disease + confidence.\n"
             "3. Follow the treatment tips below.")
    st.markdown("---")
    st.caption("Model: MobileNetV2 · ~2.24M params · 15 disease classes\n\n"
               "Computer Vision Semester Project")

st.markdown(
    """
    <div class="hero">
      <h1>🌿 Plant Leaf Disease Detector</h1>
      <p>Upload a leaf photo to instantly diagnose diseases in pepper, potato
      and tomato plants — with confidence scores and treatment advice.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, class_names, device = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

col_up, col_res = st.columns([1, 1.25], gap="large")

with col_up:
    st.subheader("📤 Upload a leaf image")
    file = st.file_uploader("Drop or browse (JPG / PNG)",
                            type=["jpg", "jpeg", "png"],
                            label_visibility="collapsed")
    if file:
        img = Image.open(io.BytesIO(file.read()))
        st.image(img, caption="Uploaded leaf", use_container_width=True)
    else:
        st.info("👆 Upload a leaf image to begin. For best results, use a "
                "close-up of a single leaf on a plain background.")

with col_res:
    st.subheader("🔬 Diagnosis")
    if not file:
        st.caption("Your result will appear here.")
    else:
        results, _ = predict(img, model, class_names, device)
        top_class, top_conf = results[0]
        info = get_info(top_class)
        healthy = info.get("healthy")

        css = "healthy" if healthy else "diseased"
        pill = ('<span class="pill pill-ok">✅ Healthy</span>' if healthy
                else '<span class="pill pill-bad">⚠️ Disease detected</span>'
                if healthy is False
                else '<span class="pill pill-bad">❓ Uncertain</span>')

        st.markdown(
            f"""
            <div class="result-card {css}">
              {pill}
              <h2 style="margin:.2rem 0;">{pretty_label(top_class)}</h2>
              <p style="margin:.1rem 0; color:#555;">Confidence:
                 <b>{top_conf*100:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(top_conf, 1.0))

        st.markdown("**Symptoms**")
        st.write(info["symptoms"])
        st.markdown('<div class="tip"><b>💊 Recommended action</b><br>'
                    f'{info["treatment"]}</div>', unsafe_allow_html=True)

        st.markdown("#### Top-3 predictions")
        df = pd.DataFrame(
            {"Disease": [pretty_label(c) for c, _ in results],
             "Confidence": [round(p * 100, 2) for _, p in results]}
        ).set_index("Disease")
        st.bar_chart(df, color="#2e7d32", horizontal=True)

        if top_conf < 0.55:
            st.warning("Low confidence — the image may be unclear or show a "
                       "plant/condition outside the 15 trained classes. Try a "
                       "sharper, well-lit photo of a single leaf.")

st.markdown("---")
st.caption("⚠️ This tool is a decision aid trained on laboratory images and "
           "covers pepper, potato and tomato only. Confirm critical diagnoses "
           "with an agricultural expert.")
