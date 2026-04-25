import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd

# ─── Page Config ───
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, #f1f8ff 0%, #fbfdff 45%, #ffffff 100%);
    }
    .hero {
        padding: 1rem 1.1rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #003049 0%, #264653 60%, #2a9d8f 100%);
        color: #f8fafc;
        margin-bottom: 1rem;
        box-shadow: 0 12px 24px rgba(0, 48, 73, 0.22);
    }
    .hero p {
        margin: 0.2rem 0 0 0;
        opacity: 0.95;
    }
    .compare-card {
        background: #f8fafc;
        border: 1px solid #dbe5ef;
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
        color: #0f172a;
    }
    .compare-title {
        color: #0b3b5f;
        font-weight: 700;
        margin: 0 0 0.25rem 0;
    }
    .small-note {
        color: #334155;
        margin: 0;
    }
    .stTabs [data-baseweb="tab"] {
        color: #0f172a;
        font-weight: 600;
    }
    .stImage img {
        max-height: 420px;
        object-fit: contain;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Title ───
st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0;">🔬 SkinGuard Multi-Model Detection</h2>
      <p>Upload or capture a skin image and compare outputs from all three YOLO models: 8s, 9s, and 10s.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Load Models ───
MODEL_PATHS = {
    "YOLO-8s": "8s-best.pt",
    "YOLO-9s": "9s-best.pt",
    "YOLO-10s": "10s-best.pt",
}

@st.cache_resource
def load_models(paths: dict[str, str]) -> dict[str, YOLO]:
    missing_models = [
        f"{name}: {path}" for name, path in paths.items() if not os.path.exists(path)
    ]

    if missing_models:
        missing_text = "\n".join(f"- {item}" for item in missing_models)
        st.error(
            "Some model weights are missing:\n\n"
            f"{missing_text}\n\n"
            "Please place all 3 files in the project folder, or update MODEL_PATHS."
        )
        st.stop()

    return {name: YOLO(path) for name, path in paths.items()}


def summarize_result(result) -> dict[str, float | int]:
    boxes = result.boxes
    confidences = [float(box.conf[0]) for box in boxes]
    count = len(confidences)
    confidence = float(max(confidences)) if count else 0.0
    return {
        "detections": count,
        "confidence": confidence,
    }

models = load_models(MODEL_PATHS)

# ─── Fixed Inference Settings ───
confidence = 0.50
image_size = 512

# ─── Input Source ───
st.subheader("📷 Upload Image")

image: Image.Image | None = None
uploaded = st.file_uploader(
    "Upload a skin image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

# ─── Run Detection ───
if image is not None:
    st.divider()
    col_orig, col_summary = st.columns([1.2, 1])

    with col_orig:
        st.image(image, caption="Input Image", width="content")

    model_outputs: dict[str, dict] = {}

    with st.spinner("Running all 3 models..."):
        image_np = np.array(image)
        for model_name, model in models.items():
            results = model.predict(
                source=image_np,
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )
            result = results[0]
            annotated_bgr = result.plot()
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            model_outputs[model_name] = {
                "result": result,
                "annotated": annotated_rgb,
                "summary": summarize_result(result),
            }

    best_model = max(
        model_outputs.items(),
        key=lambda item: (
            item[1]["summary"]["detections"],
            item[1]["summary"]["confidence"],
        ),
    )[0]

    with col_summary:
        st.metric("🏆 Most Active Model", best_model)
        st.markdown(
            """
            <div class="compare-card">
              <p class="compare-title">How Ranking Works</p>
                            <p class="small-note">Higher detections wins. If equal, higher confidence wins.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        total_detections = sum(
            int(output["summary"]["detections"])
            for output in model_outputs.values()
        )
        st.metric("Total Detections (All Models)", total_detections)

    st.subheader("🧪 Model Output Comparison")
    model_cols = st.columns(len(model_outputs))

    for col, (model_name, output) in zip(model_cols, model_outputs.items()):
        summary = output["summary"]
        result = output["result"]

        with col:
            st.markdown(f"### {model_name}")
            st.image(
                output["annotated"],
                caption=f"{model_name} Detection Result",
                width="content",
            )
            m1, m2 = st.columns(2)
            m1.metric("Detections", int(summary["detections"]))
            m2.metric("Confidence", f"{summary['confidence']:.2%}")

    st.subheader("🧾 Detection Details")
    detail_tabs = st.tabs(list(model_outputs.keys()))

    for tab, model_name in zip(detail_tabs, model_outputs):
        output = model_outputs[model_name]
        summary = output["summary"]
        result = output["result"]

        with tab:
            if summary["detections"] == 0:
                st.warning("No lesions detected by this model.")
            else:
                for i, box in enumerate(result.boxes):
                    conf_val = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = result.names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    with st.expander(
                        f"Detection {i+1} - {cls_name} ({conf_val:.1%})"
                    ):
                        st.markdown(f"- **Class:** {cls_name}")
                        st.markdown(f"- **Confidence:** {conf_val:.2%}")
                        st.markdown(
                            f"- **Bounding Box:** "
                            f"({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})"
                        )

    st.subheader("📊 Numeric Comparison")
    comparison_rows = []
    for model_name, output in model_outputs.items():
        summary = output["summary"]
        comparison_rows.append(
            {
                "Model": model_name,
                "Detections": int(summary["detections"]),
                "ConfidenceValue": float(summary["confidence"]),
                "Confidence": f"{summary['confidence']:.2%}",
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["Detections", "ConfidenceValue"], ascending=False
    )
    comparison_df = comparison_df.drop(columns=["ConfidenceValue"])
    st.dataframe(comparison_df, width="stretch", hide_index=True)

    # ─── Disclaimer ───
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This tool is for educational/research purposes only. "
        "It is NOT a substitute for professional medical diagnosis. "
        "Always consult a qualified dermatologist."
    )
else:
    st.info("👆 Upload an image or take a photo to get started.")
