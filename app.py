"""
EcoSort – Waste Classification Using Image Recognition
Flask Backend with PyTorch MobileNetV2 Inference Engine

Department of Electronics and Communication Engineering (ECE) – 5th Semester
Academic Computer Vision Project
"""

import os
import io
import time
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, request, jsonify, render_template
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ecosort")

# ---------------------------------------------------------------------------
# Application Constants
# ---------------------------------------------------------------------------
MAX_CONTENT_BYTES = 5 * 1024 * 1024       # 5 MB hard limit
CONFIDENCE_THRESHOLD = 0.40               # Below this → "Unknown / Uncertain"
MODEL_PATH = os.path.join("models", "ecosort_mobilenetv2.pth")
DEVICE = torch.device("cpu")             # Render free-tier is CPU-only

# Waste categories aligned with model training labels (train.py)
CATEGORY_LABELS = [
    "Cardboard",        # idx 0
    "Electronic Waste", # idx 1
    "Glass",            # idx 2
    "Metal",            # idx 3
    "Organic Waste",    # idx 4
    "Paper",            # idx 5
    "Plastic",          # idx 6
    "Trash",            # idx 7  → mapped to Unknown / Uncertain
]

# Cardboard → Paper (both cellulosic, same handling advice)
# Trash     → Unknown / Uncertain
LABEL_MAP = {
    "Cardboard":        "Paper",
    "Electronic Waste": "Electronic Waste",
    "Glass":            "Glass",
    "Metal":            "Metal",
    "Organic Waste":    "Organic Waste",
    "Paper":            "Paper",
    "Plastic":          "Plastic",
    "Trash":            "Unknown / Uncertain",
}

# Disposal guidance per mapped category
DISPOSAL_GUIDANCE = {
    "Plastic": [
        "Empty and thoroughly rinse all containers before disposal.",
        "Check the resin identification code (1-7) against your local recycling programme.",
        "Soft plastic films and bags require separate supermarket drop-off collection.",
        "Crush rigid bottles to save space in collection bins.",
    ],
    "Paper": [
        "Recycle clean, dry paper, cardboard, and paperboard where accepted.",
        "Flatten corrugated boxes to maximise bin capacity.",
        "Do NOT recycle grease-soaked or food-contaminated paper (e.g. pizza boxes).",
        "Remove excessive packaging tape and plastic wrapping before recycling.",
    ],
    "Metal": [
        "Rinse food and beverage cans thoroughly to prevent contamination.",
        "Aluminium cans are infinitely recyclable with no quality degradation.",
        "Wrap sharp can lids safely or keep them inside the crushed can.",
        "Take large scrap metal to a dedicated municipal drop-off facility.",
    ],
    "Glass": [
        "Separate glass bottles and jars from other waste streams.",
        "Rinse containers and remove metal/plastic caps before recycling.",
        "Do NOT mix ceramics, Pyrex, mirrors, or windowpanes with container glass.",
        "Wrap broken glass securely in paper before placing in general waste.",
    ],
    "Organic Waste": [
        "Divert to home composting or municipal organic/wet waste collection bins.",
        "Avoid placing dairy, meat, or cooked oils into a standard backyard compost heap.",
        "Compostable packaging must be certified; standard plastic bags contaminate compost.",
        "Composting organic matter prevents potent methane emissions in landfill.",
    ],
    "Electronic Waste": [
        "NEVER discard batteries or electronics in standard household or recycling bins.",
        "Use authorised e-waste collection centres or manufacturer take-back schemes.",
        "Tape exposed lithium battery terminals with non-conductive tape before transport.",
        "Perform a factory reset to protect personal data before disposal.",
    ],
    "Unknown / Uncertain": [
        "The model could not identify this item with sufficient confidence.",
        "Re-photograph with a single waste item, clear background, and good lighting.",
        "Check the item's label for sorting codes or material symbols.",
        "Consult your local municipal waste authority for guidance.",
    ],
}

# Allowed image MIME types
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

# ---------------------------------------------------------------------------
# Flask Application Initialisation
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_BYTES

# ---------------------------------------------------------------------------
# Model Loading (singleton – loaded once at startup)
# ---------------------------------------------------------------------------
_model = None
_model_ready = False


def build_model(num_classes: int) -> nn.Module:
    """Create MobileNetV2 with a custom classification head matching training."""
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features, num_classes),
    )
    return model


def load_model() -> bool:
    """Load trained weights from disk into the global model singleton."""
    global _model, _model_ready

    if not os.path.exists(MODEL_PATH):
        log.warning(
            "Model weights not found at '%s'. Run 'python train.py' first to train "
            "and save the model. Inference will be unavailable until the model is "
            "present.", MODEL_PATH
        )
        return False

    try:
        log.info("Loading model weights from '%s' ...", MODEL_PATH)
        t0 = time.perf_counter()

        model = build_model(num_classes=len(CATEGORY_LABELS))
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        model.to(DEVICE)

        _model = model
        _model_ready = True
        log.info("Model loaded in %.2f s - ready for inference.", time.perf_counter() - t0)
        return True

    except Exception as exc:
        log.error("Failed to load model: %s", exc, exc_info=True)
        _model_ready = False
        return False


# ImageNet-normalisation preprocessing (matches training transforms)
_inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def run_inference(image: Image.Image):
    """
    Run a forward pass through MobileNetV2 for the given PIL image.

    Returns
    -------
    category : str   – Mapped human-readable category name.
    confidence : float – Softmax probability for the predicted class.
    uncertain : bool – True when confidence is below CONFIDENCE_THRESHOLD.
    """
    if not _model_ready or _model is None:
        raise RuntimeError("Model is not loaded. Run train.py to generate model weights.")

    # Convert to RGB (handles RGBA / grayscale images safely)
    image = image.convert("RGB")
    tensor = _inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    confidence, class_idx = probabilities.max(dim=0)
    confidence = confidence.item()
    class_idx = class_idx.item()

    raw_label = CATEGORY_LABELS[class_idx]
    mapped_category = LABEL_MAP.get(raw_label, "Unknown / Uncertain")

    uncertain = (confidence < CONFIDENCE_THRESHOLD) or (mapped_category == "Unknown / Uncertain")

    if uncertain:
        mapped_category = "Unknown / Uncertain"

    return mapped_category, confidence, uncertain


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main EcoSort single-page application."""
    return render_template("index.html")


@app.route("/health")
def health():
    """
    Lightweight health-check endpoint for Render uptime monitoring.
    Returns model readiness without exposing internal details.
    """
    return jsonify({
        "status": "ok",
        "service": "EcoSort Waste Classification API",
        "model_ready": _model_ready,
        "model_path": MODEL_PATH,
        "categories_supported": len(CATEGORY_LABELS),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict

    Accepts a multipart/form-data request with an 'image' field.
    Returns a JSON object with the predicted waste category, confidence
    score, disposal guidance, and uncertainty flag.
    """
    # 1. Model availability check
    if not _model_ready:
        return jsonify({
            "success": False,
            "error": (
                "The classification model is not loaded. "
                "Please run 'python train.py' to train the model, then restart the server."
            ),
        }), 503

    # 2. File presence validation
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in the request."}), 400

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify({"success": False, "error": "The uploaded file has an empty filename."}), 400

    # 3. MIME-type validation (check actual content-type, not just extension)
    content_type = uploaded_file.content_type or ""
    ext = os.path.splitext(uploaded_file.filename)[1].lower()
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    if content_type.lower() not in ALLOWED_MIME and ext not in allowed_extensions:
        return jsonify({
            "success": False,
            "error": "Unsupported file format. Please upload a JPEG, PNG, or WEBP image.",
        }), 415

    # 4. Read image bytes
    try:
        image_bytes = uploaded_file.read()
    except Exception:
        return jsonify({"success": False, "error": "Failed to read the uploaded file."}), 400

    if len(image_bytes) == 0:
        return jsonify({"success": False, "error": "The uploaded file is empty."}), 400

    if len(image_bytes) > MAX_CONTENT_BYTES:
        return jsonify({
            "success": False,
            "error": f"File size exceeds the {MAX_CONTENT_BYTES // (1024*1024)} MB limit.",
        }), 413

    # 5. Decode the image safely (reject corrupt / invalid files)
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image.verify()                       # Detect truncated/corrupt files
        pil_image = Image.open(io.BytesIO(image_bytes))  # Re-open after verify
        pil_image.load()                          # Force full decode
    except UnidentifiedImageError:
        return jsonify({
            "success": False,
            "error": "The file could not be identified as a valid image.",
        }), 422
    except Exception:
        return jsonify({
            "success": False,
            "error": "The image file appears to be corrupt or incomplete.",
        }), 422

    # 6. Run model inference
    try:
        category, confidence, uncertain = run_inference(pil_image)
    except RuntimeError as exc:
        return jsonify({"success": False, "error": str(exc)}), 503
    except Exception:
        log.exception("Unexpected error during inference")
        return jsonify({
            "success": False,
            "error": "An internal error occurred during image classification.",
        }), 500

    # 7. Build response
    guidance = DISPOSAL_GUIDANCE.get(category, DISPOSAL_GUIDANCE["Unknown / Uncertain"])

    log.info(
        "Prediction: %s | Confidence: %.2f%% | Uncertain: %s | File: %s",
        category, confidence * 100, uncertain, uploaded_file.filename,
    )

    return jsonify({
        "success": True,
        "category": category,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "uncertain": uncertain,
        "disposal_guidance": guidance,
    }), 200


# ---------------------------------------------------------------------------
# Error Handlers (no stack traces exposed to clients)
# ---------------------------------------------------------------------------

@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_):
    return jsonify({
        "success": False,
        "error": f"Upload rejected: file exceeds the {MAX_CONTENT_BYTES // (1024*1024)} MB size limit.",
    }), 413


@app.errorhandler(404)
def handle_404(_):
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(405)
def handle_405(_):
    return jsonify({"success": False, "error": "Method not allowed."}), 405


@app.errorhandler(500)
def handle_500(_):
    return jsonify({"success": False, "error": "Internal server error."}), 500


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
# Load model when the module is imported (Gunicorn worker startup).
# This guarantees inference is available without the first-request delay.
load_model()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    log.info("Starting EcoSort dev server on http://127.0.0.1:%d  (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)

