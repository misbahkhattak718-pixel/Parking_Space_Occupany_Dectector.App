"""
Parking Space Occupancy Detector
==================================
A Streamlit web app for real-time detection and classification of parking space occupancy.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Lazy import cv2 to avoid deployment issues ──────────────────────────────────
def get_cv2():
    import cv2
    return cv2

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "best.pt"

# ── Page config (must be the very first Streamlit call) ────────────────────────
st.set_page_config(
    page_title = "Parking Space Occupancy Detector",
    page_icon  = "🅿️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS Styling ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 12px 16px;
    }
    .stProgress > div > div { border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
CLASS_NAMES: Dict[int, str] = {
    0: "free",
    1: "occupied",
    2: "partially_occupied",
}

BOX_COLOUR: Dict[int, tuple] = {
    0: (0,   200,   0),     # green  — free
    1: (0,     0, 220),     # red    — occupied
    2: (0,   210, 230),     # yellow — partially occupied
}

TEXT_COLOUR: Dict[int, tuple] = {
    0: (255, 255, 255),     # white on green
    1: (255, 255, 255),     # white on red
    2: (0,     0,   0),     # black on yellow
}

MAX_DIM = 1920


# ── ParkingDetector Class ──────────────────────────────────────────────────────

class ParkingDetector:
    """Load a YOLOv8 model and run parking-space occupancy detection."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.50,
        iou_threshold:  float = 0.45,
    ) -> None:
        self.model_path     = Path(model_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold  = float(iou_threshold)
        self._model         = None
        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_path}\n\n"
                "To train the model locally, run:\n"
                "  python convert_dataset.py\n"
                "  python train.py\n\n"
                "Or download pre-trained model from GitHub releases."
            )
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Run:  pip install ultralytics"
            ) from e
        self._model = YOLO(str(self.model_path))

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """Return a copy of *image* that is 3-channel uint8 RGB."""
        cv2 = get_cv2()
        img = image.copy()

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3:
            if img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        return img

    def detect(
        self,
        image: np.ndarray,
        conf: float | None = None,
        iou:  float | None = None,
    ) -> List[Dict[str, Any]]:
        """Run inference on a single RGB image."""
        cv2 = get_cv2()
        if self._model is None:
            raise RuntimeError("Model is not loaded.")

        conf = conf if conf is not None else self.conf_threshold
        iou  = iou  if iou  is not None else self.iou_threshold

        img_rgb = self._preprocess(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        results = self._model.predict(
            source  = img_bgr,
            conf    = conf,
            iou     = iou,
            verbose = False,
        )

        detections: List[Dict[str, Any]] = []
        for r in results:
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                xyxy  = r.boxes.xyxy[i].cpu().numpy().tolist()
                cid   = int(r.boxes.cls[i].cpu().numpy())
                score = float(r.boxes.conf[i].cpu().numpy())
                detections.append(
                    {
                        "bbox":       [int(v) for v in xyxy],
                        "class_id":   cid,
                        "class_name": CLASS_NAMES.get(cid, f"class_{cid}"),
                        "confidence": round(score, 4),
                    }
                )

        return detections

    def draw_boxes(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of *image* (RGB)."""
        cv2 = get_cv2()
        img = self._preprocess(image)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cid   = det["class_id"]
            cname = det["class_name"]
            score = det["confidence"]

            colour     = BOX_COLOUR.get(cid,   (180, 180, 180))
            txt_colour = TEXT_COLOUR.get(cid,  (255, 255, 255))

            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

            label = f"{cname} {score * 100:.1f}%" if show_confidence else cname

            font      = cv2.FONT_HERSHEY_SIMPLEX
            fscale    = 0.48
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, fscale, thickness)

            pad   = 3
            lx1   = x1
            ly1   = max(y1 - th - baseline - pad * 2, 0)
            lx2   = x1 + tw + pad * 2
            ly2   = max(y1, ly1 + th + baseline + pad * 2)

            cv2.rectangle(img, (lx1, ly1), (lx2, ly2), colour, cv2.FILLED)
            cv2.putText(
                img,
                label,
                (lx1 + pad, ly2 - baseline - pad),
                font,
                fscale,
                txt_colour,
                thickness,
                cv2.LINE_AA,
            )

        return img

    def get_summary(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarise a list of detections."""
        total    = len(detections)
        free     = sum(1 for d in detections if d["class_id"] == 0)
        occupied = sum(1 for d in detections if d["class_id"] == 1)
        partial  = sum(1 for d in detections if d["class_id"] == 2)
        rate     = (occupied + partial) / total if total > 0 else 0.0

        return {
            "total":          total,
            "free_count":     free,
            "occupied_count": occupied,
            "partial_count":  partial,
            "occupancy_rate": round(rate, 4),
        }


# ── Helper Functions ──────────────────────────────────────────────────────────

def pil_to_rgb(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img.convert("RGB"), dtype=np.uint8)


def rgb_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


# ── Cached model loader ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model weights…")
def load_detector(model_path: str, conf: float, iou: float):
    """Returns (detector, error_string | None)."""
    try:
        # Check if model exists
        if not Path(model_path).exists():
            return None, f"Model file not found at {model_path}"
        
        det = ParkingDetector(model_path, conf_threshold=conf, iou_threshold=iou)
        return det, None
    except FileNotFoundError as exc:
        return None, str(exc)
    except ImportError as exc:
        return None, f"Import error: {exc}"
    except Exception as exc:
        return None, f"Error: {exc}"


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value = 0.10,
        max_value = 1.00,
        value     = 0.50,
        step      = 0.05,
        help      = "Detections below this score are discarded.",
    )

    iou_threshold = st.slider(
        "NMS IoU Threshold",
        min_value = 0.10,
        max_value = 1.00,
        value     = 0.45,
        step      = 0.05,
        help      = "Non-Maximum Suppression overlap threshold.",
    )

    show_conf = st.toggle("Show Confidence Scores on Boxes", value=True)

    st.divider()

    st.markdown(
        """
        **Colour legend**
        🟩 **Green** — Free  
        🟥 **Red** — Occupied  
        🟨 **Yellow** — Partially Occupied
        """
    )

    st.divider()
    st.caption(f"Model: `{MODEL_PATH.name}`")
    st.caption(f"Location: `{MODEL_PATH}`")
    
    # Check model status
    if MODEL_PATH.exists():
        file_size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        st.success(f"✅ Model loaded ({file_size_mb:.2f} MB)")
    else:
        st.error(
            "❌ **Model Not Found**\n\n"
            "The model file is missing. You have two options:\n\n"
            "**Option 1: Local Development**\n"
            "Train the model:\n"
            "```bash\n"
            "python convert_dataset.py\n"
            "python train.py\n"
            "```\n\n"
            "**Option 2: Deployment**\n"
            "Ensure `models/best.pt` is included in:\n"
            "- Git repository (via Git LFS)\n"
            "- Deployment package\n"
            "- Or download from releases"
        )


# ── Main page ──────────────────────────────────────────────────────────────────

st.title("🅿️ Parking Space Occupancy Detector")
st.markdown(
    "Upload a parking lot image and click **Detect** to classify each space "
    "as *free*, *occupied*, or *partially occupied*."
)

uploaded = st.file_uploader(
    "Choose an image",
    type    = ["jpg", "jpeg", "png"],
    label_visibility = "collapsed",
)

detect_btn = st.button(
    "🔍 Detect",
    type     = "primary",
    disabled = (uploaded is None),
)

# ── Detection flow ────────────────────────────────────────────────────────────

if uploaded is not None:

    pil_image    = Image.open(uploaded)
    original_rgb = pil_to_rgb(pil_image)

    # Show the image before detection is triggered
    if not detect_btn:
        st.image(original_rgb, caption="Uploaded image", use_container_width=True)

    else:
        # Load model
        detector, err = load_detector(str(MODEL_PATH), conf_threshold, iou_threshold)

        if err:
            if "not found" in err.lower():
                st.error(
                    "🚫 **Model File Not Found**\n\n"
                    "This app requires the model file: `models/best.pt`\n\n"
                    "**For Local Use:**\n"
                    "```bash\n"
                    "python convert_dataset.py\n"
                    "python train.py\n"
                    "```\n\n"
                    "**For Deployment:**\n"
                    "- Ensure `models/best.pt` is in your repository\n"
                    "- Use Git LFS to track large files\n"
                    "- Or upload to cloud storage and download on startup"
                )
            else:
                st.error(f"⚠️ Could not load model:\n\n{err}")
            st.stop()

        # Run inference
        with st.spinner("Running detection…"):
            try:
                detections = detector.detect(
                    original_rgb,
                    conf = conf_threshold,
                    iou  = iou_threshold,
                )
                annotated = detector.draw_boxes(
                    original_rgb,
                    detections,
                    show_confidence = show_conf,
                )
            except Exception as exc:
                st.error(f"Detection failed: {exc}")
                st.stop()

        # ── Warning if nothing found ──────────────────────────────────────────
        if not detections:
            st.warning(
                "⚠️ No parking spaces detected in this image. "
                "Try lowering the **Confidence Threshold** in the sidebar, "
                "or check that the image shows a parking lot."
            )

        # ── Side-by-side images ───────────────────────────────────────────────
        col_l, col_r = st.columns(2, gap="medium")
        with col_l:
            st.subheader("📷 Original")
            st.image(original_rgb, use_container_width=True)
        with col_r:
            st.subheader("🔍 Detected Spaces")
            st.image(annotated, use_container_width=True)

        # ── Summary metrics ───────────────────────────────────────────────────
        summary = detector.get_summary(detections)

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📊 Total Spaces", summary["total"])
        m2.metric("✅ Free",          summary["free_count"])
        m3.metric("🚗 Occupied",      summary["occupied_count"])
        m4.metric("⚠️ Partial",       summary["partial_count"])

        occ_pct = summary["occupancy_rate"] * 100
        st.markdown(f"**Occupancy Rate: {occ_pct:.1f}%**")
        st.progress(float(summary["occupancy_rate"]))

        # ── Detections table ──────────────────────────────────────────────────
        if detections:
            st.divider()
            st.subheader("📋 Detection Details")

            STATUS_DISPLAY = {
                "free":                 "✅ Free",
                "occupied":             "🚗 Occupied",
                "partially_occupied":   "⚠️ Partial",
            }

            rows = []
            for idx, d in enumerate(detections, start=1):
                x1, y1, x2, y2 = d["bbox"]
                rows.append(
                    {
                        "Space #":     idx,
                        "Status":      STATUS_DISPLAY.get(d["class_name"],
                                                          d["class_name"]),
                        "Confidence":  f"{d['confidence'] * 100:.1f}%",
                        "X1":          x1,
                        "Y1":          y1,
                        "X2":          x2,
                        "Y2":          y2,
                        "W (px)":      x2 - x1,
                        "H (px)":      y2 - y1,
                    }
                )

            df = pd.DataFrame(rows)

            def _row_colour(row):
                val = row["Status"]
                if "Free"    in val: bg = "#1a4a1a"; fg = "#afffaf"
                elif "Occup" in val: bg = "#4a1a1a"; fg = "#ffafaf"
                else:                bg = "#4a4a1a"; fg = "#ffffaf"
                return [f"background-color:{bg}; color:{fg}"
                        if c == "Status" else "" for c in row.index]

            st.dataframe(
                df.style.apply(_row_colour, axis=1),
                use_container_width = True,
                hide_index          = True,
            )

        # ── Download ──────────────────────────────────────────────────────────


        st.divider()
        st.download_button(
            label     = "⬇️ Download Annotated Image",
            data      = rgb_to_bytes(annotated),
            file_name = "parking_detection_result.png",
            mime      = "image/png",
        )
