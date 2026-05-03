import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import gdown
import os

from model import build_model

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Blood Cell Classifier", layout="wide")

# -----------------------------------
# UI CSS
# -----------------------------------
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 16px !important; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px; }
h1 { text-align: center; font-size: 2.5rem !important; }
.subtitle { text-align: center; color: gray; }
.result-box {
    background: #111827;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    color: #00ffcc;
    font-size: 1.3rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------
st.markdown("<h1>🧬 Blood Cell Classifier</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Based Multi-Cell Detection System</div>', unsafe_allow_html=True)

# -----------------------------------
# DOWNLOAD MODEL
# -----------------------------------
MODEL_PATH = "model_best.pt"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1OO3Uh4O5gWprhlXeqfeBNeLI_nKbbWfH"
    gdown.download(url, MODEL_PATH, quiet=False)

# -----------------------------------
# LABELS
# -----------------------------------
labels = ["RBC", "WBC", "Platelets"]

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = build_model(len(labels))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# -----------------------------------
# INPUT
# -----------------------------------
uploaded = st.file_uploader("📤 Upload Blood Cell Image")

# -----------------------------------
# PROCESS
# -----------------------------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Validate image
    x_full = tfm(img).unsqueeze(0)
    with torch.no_grad():
        probs_full = torch.softmax(model(x_full), dim=1)

    if probs_full.max().item() < 0.6:
        st.error("❌ Not a valid blood cell image")
        st.stop()

    # Sliding window
    img_np = np.array(img)
    h, w, _ = img_np.shape

    patch_size = 64
    stride = 32

    detected = set()

    thresholds = {
        "RBC": 0.55,
        "WBC": 0.6,
        "Platelets": 0.75
    }

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_np[y:y+patch_size, x:x+patch_size]

            if np.mean(patch) > 240:
                continue

            patch_img = Image.fromarray(patch)
            x_tensor = tfm(patch_img).unsqueeze(0)

            with torch.no_grad():
                probs = torch.softmax(model(x_tensor), dim=1)
                pred = probs.argmax(1).item()

            label = labels[pred]
            confidence = probs[0, pred].item()

            if confidence > thresholds.get(label, 0.5):
                detected.add(label)

    # -----------------------------------
    # OUTPUT
    # -----------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(img, caption="🖼 Uploaded Image", use_column_width=True)

    with col2:
        if detected:
            st.markdown(
                f'<div class="result-box">🔬 Detected Cells:<br>{", ".join(sorted(detected))}</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("⚠️ No cells detected")
