import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import os
import pandas as pd

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

st.markdown(
    """
    <div class="hero">
      <h2 style="margin:0;">🔬 SkinGuard Multi-Model Detection</h2>
            <p>Upload or capture a skin image and compare outputs from all six YOLO models: 8s, 8m, 8l, 9s, 9m, and 10s.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

MODEL_PATHS = {
    "YOLO-8s": "8s-best.pt",
    "YOLO-8m": "8m-best.pt",
    "YOLO-8l": "8l-best.pt",
    "YOLO-9s": "9s-best.pt",
    "YOLO-9m": "9m-best.pt",
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
            "Please place all required files in the project folder, or update MODEL_PATHS."
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


def render_on_original(image: Image.Image, result) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for box in result.boxes:
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        draw.rectangle([x1, y1, x2, y2], outline="#0066ff", width=3)

        label = f"{conf_val:.2%}"
        text_bbox = draw.textbbox((x1, y1), label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x1 = x1
        text_y1 = max(0, y1 - text_height - 6)
        text_x2 = text_x1 + text_width + 8
        text_y2 = text_y1 + text_height + 4

        draw.rectangle([text_x1, text_y1, text_x2, text_y2], fill="#0066ff")
        draw.text((text_x1 + 4, text_y1 + 2), label, fill="white")

    return annotated

models = load_models(MODEL_PATHS)

confidence = 0.50
image_size = 512

st.subheader("📷 Upload Image")

image: Image.Image | None = None
uploaded = st.file_uploader(
    "Upload a skin image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

if image is not None:
    st.divider()
    col_orig, col_summary = st.columns([1.2, 1])

    with col_orig:
        st.image(image, caption="Input Image", width="content")

    model_outputs: dict[str, dict] = {}

    with st.spinner("Running all 6 models..."):
        image_np = np.array(image)
        for model_name, model in models.items():
            results = model.predict(
                source=image_np,
                conf=confidence,
                imgsz=image_size,
                verbose=False,
            )
            result = results[0]
            annotated_image = render_on_original(image, result)
            model_outputs[model_name] = {
                "result": result,
                "annotated": annotated_image,
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
        st.metric(f"Total Detections (All {len(models)} Models)", total_detections)

    st.subheader("🧪 Model Output Comparison")
    model_items = list(model_outputs.items())
    columns_per_row = 3

    for row_start in range(0, len(model_items), columns_per_row):
        row_models = model_items[row_start : row_start + columns_per_row]
        row_cols = st.columns(columns_per_row)

        for col, (model_name, output) in zip(row_cols, row_models):
            summary = output["summary"]

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

    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This tool is for educational/research purposes only. "
        "It is NOT a substitute for professional medical diagnosis. "
        "Always consult a qualified dermatologist."
    )
else:
    st.info("👆 Upload an image or take a photo to get started.")
