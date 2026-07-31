"""
Potato Disease Detection System — Vercel Serverless Deployment
-----------------------------------------------------------------
This version uses a TensorFlow Lite model + the lightweight `ai-edge-litert`
interpreter instead of full TensorFlow. Full TensorFlow installs to ~950 MB
(over Vercel's 500 MB serverless function limit); the TFLite interpreter
installs to a few dozen MB, so the whole app comfortably fits.

The model was converted from best_MobileNet.keras -> best_MobileNet.tflite
using dynamic-range quantization. Verified against the original model on
20 test images: 20/20 identical top predictions, max probability
difference 0.09 (does not change any classification outcome).

Uploaded images are processed entirely in memory (serverless filesystems
are read-only/ephemeral) and the preview image is returned to the browser
as a base64 data URI.
"""

import os
import logging
import numpy as np
import cv2
import base64
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, request, jsonify, render_template

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_MobileNet.tflite")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB max upload size

CLASS_NAMES = [
    "Potato___bacterial_wilt",
    "Potato___early_blight",
    "Potato___healthy",
    "Potato___late_blight",
    "Potato___leafroll_virus",
    "Potato___mosaic_virus",
    "Potato___pests",
    "Potato___phytophthora",
]

# --------------------------------------------------------------------------
# Dosage calculation settings
# --------------------------------------------------------------------------
# The app auto-calculates "Dosage for 30 Decimal Land" from each disease's
# mixing ratio (ml per liter of water). This requires an assumption about
# how many liters of spray solution are typically applied per acre with a
# standard knapsack sprayer for potato crops. 200 L/acre is a commonly used
# extension-service figure. ADJUST THIS if your local agricultural office
# recommends a different spray volume.
SPRAY_VOLUME_PER_ACRE_LITERS = 200
DECIMAL_TO_ACRE = 0.01          # 1 decimal = 1/100 acre
TARGET_LAND_DECIMAL = 30        # "30 Decimal Land" as required


def calculate_spray_volume_liters(decimal_land: float = TARGET_LAND_DECIMAL) -> float:
    """Total liters of spray solution needed to cover the given land area."""
    acres = decimal_land * DECIMAL_TO_ACRE
    return round(acres * SPRAY_VOLUME_PER_ACRE_LITERS, 1)


def calculate_dosage_text(min_ml_per_l, max_ml_per_l, medicine_name: str) -> str:
    """
    Build a human-readable dosage string for TARGET_LAND_DECIMAL, computed
    automatically from the disease's mixing ratio and the standard spray
    volume assumption above.
    """
    if min_ml_per_l is None:
        return None

    spray_volume = calculate_spray_volume_liters()
    min_total = min_ml_per_l * spray_volume
    max_total = max_ml_per_l * spray_volume

    if round(min_total) == round(max_total):
        amount_text = f"{min_total:.0f} ml"
    else:
        amount_text = f"{min_total:.0f}–{max_total:.0f} ml"

    return (
        f"{amount_text} of {medicine_name} mixed into {spray_volume:.0f} liters "
        f"of water (covers {TARGET_LAND_DECIMAL} Decimal land)"
    )


DEFAULT_IMPORTANT_NOTE = (
    "Always follow the pesticide/fungicide label instructions and consult "
    "your local agricultural extension officer before applying any chemical."
)

# --------------------------------------------------------------------------
# Disease recommendation database
# --------------------------------------------------------------------------
# NOTE ON DOSAGE: "mixing_ratio_min_ml_per_l" / "mixing_ratio_max_ml_per_l"
# drive the automatic dosage calculation above. Set both to None when no
# numeric mixing ratio applies (e.g. no chemical treatment exists, or the
# product varies too much to give one figure).
CLASS_INFO = {
    "Potato___healthy": {
        "display_name": "Healthy Plant",
        "status": "healthy",
        "scientific_name": None,
        "description": "No disease detected. The leaf appears healthy.",
        "medicine": "No pesticide or fungicide is required.",
        "mixing_ratio_text": None,
        "mixing_ratio_min_ml_per_l": None,
        "mixing_ratio_max_ml_per_l": None,
        "treatment": [
            "Continue regular field monitoring.",
            "Maintain proper irrigation.",
            "Apply balanced fertilizer as per normal schedule.",
        ],
        "prevention": [
            "Keep monitoring the crop weekly for early signs of disease.",
            "Maintain good field hygiene and drainage.",
        ],
        "important_note": "No pesticide or fungicide is required at this time.",
    },
    "Potato___early_blight": {
        "display_name": "Early Blight",
        "status": "disease",
        "scientific_name": "Alternaria solani",
        "description": "A fungal disease causing dark concentric-ring spots on "
                        "older leaves, often leading to premature leaf drop.",
        "medicine": "Triazole or Strobilurin fungicide",
        "mixing_ratio_text": "0.5–1 ml per liter of water",
        "mixing_ratio_min_ml_per_l": 0.5,
        "mixing_ratio_max_ml_per_l": 1.0,
        "treatment": [
            "Remove and destroy infected leaves.",
            "Spray fungicide uniformly over the plant.",
            "Repeat spraying after 7–10 days.",
        ],
        "prevention": [
            "Practice crop rotation.",
            "Remove plant debris after harvest.",
            "Avoid excessive humidity around the plants.",
        ],
        "important_note": DEFAULT_IMPORTANT_NOTE,
    },
    "Potato___late_blight": {
        "display_name": "Late Blight",
        "status": "disease",
        "scientific_name": "Phytophthora infestans",
        "description": "A fast-spreading disease causing dark, water-soaked "
                        "lesions; historically responsible for the Irish potato famine.",
        "medicine": "Copper-based fungicide",
        "mixing_ratio_text": "2–3 ml per liter of water",
        "mixing_ratio_min_ml_per_l": 2.0,
        "mixing_ratio_max_ml_per_l": 3.0,
        "treatment": [
            "Remove and destroy severely infected plants.",
            "Improve field drainage.",
            "Spray fungicide immediately upon detection.",
        ],
        "prevention": [
            "Avoid waterlogging in the field.",
            "Ensure good air circulation between plants.",
        ],
        "important_note": DEFAULT_IMPORTANT_NOTE,
    },
    "Potato___bacterial_wilt": {
        "display_name": "Bacterial Wilt",
        "status": "disease",
        "scientific_name": "Ralstonia solanacearum",
        "description": "A bacterial infection that causes wilting, yellowing, "
                        "and eventual collapse of the plant due to vascular blockage.",
        "medicine": "No effective curative chemical treatment exists.",
        "mixing_ratio_text": None,
        "mixing_ratio_min_ml_per_l": None,
        "mixing_ratio_max_ml_per_l": None,
        "treatment": [
            "Remove and destroy infected plants immediately.",
            "Sterilize tools after use on infected plants.",
            "Improve field drainage.",
            "Apply bleaching powder to infected soil if appropriate for your region.",
        ],
        "prevention": [
            "Use certified disease-free seed tubers.",
            "Practice crop rotation with non-host crops.",
            "Avoid working in wet fields to limit bacterial spread.",
        ],
        "important_note": "There is no curative chemical for bacterial wilt — "
                           "prevention and sanitation are the primary controls. "
                           + DEFAULT_IMPORTANT_NOTE,
    },
    "Potato___leafroll_virus": {
        "display_name": "Leafroll Virus",
        "status": "disease",
        "scientific_name": "Potato Leafroll Virus (PLRV)",
        "description": "A viral infection causing upward rolling and stiffening "
                        "of leaves, often paired with stunted growth. Spread mainly by aphids.",
        "medicine": "Systemic insecticide against aphids (the virus carrier)",
        "mixing_ratio_text": "0.5 ml per liter of water",
        "mixing_ratio_min_ml_per_l": 0.5,
        "mixing_ratio_max_ml_per_l": 0.5,
        "treatment": [
            "Control aphid populations with systemic insecticide.",
            "Remove and destroy infected plants.",
            "Use certified virus-free seed tubers for future planting.",
        ],
        "prevention": [
            "Use certified seed tubers.",
            "Monitor and control aphid populations early in the season.",
        ],
        "important_note": "There is no direct cure for the virus itself — "
                           "control focuses on the aphid vector. " + DEFAULT_IMPORTANT_NOTE,
    },
    "Potato___mosaic_virus": {
        "display_name": "Mosaic Virus",
        "status": "disease",
        "scientific_name": "Potato Virus Y / Potato Virus X (exact strain requires lab confirmation)",
        "description": "A viral infection producing a mottled light/dark green "
                        "mosaic pattern on leaves and reduced yield. Spread mainly by aphids.",
        "medicine": "No direct cure — control insect (aphid) vectors.",
        "mixing_ratio_text": None,
        "mixing_ratio_min_ml_per_l": None,
        "mixing_ratio_max_ml_per_l": None,
        "treatment": [
            "Remove and destroy infected plants.",
            "Use certified virus-free seed for future planting.",
            "Control aphid populations to limit spread.",
        ],
        "prevention": [
            "Use certified virus-free seed tubers.",
            "Control aphids early in the growing season.",
        ],
        "important_note": "No direct cure exists for mosaic virus. " + DEFAULT_IMPORTANT_NOTE,
    },
    "Potato___pests": {
        "display_name": "Pest Damage",
        "status": "disease",
        "scientific_name": None,
        "description": "Visible leaf damage caused by insect pests rather than "
                        "a pathogen (e.g. chewing or sucking insects).",
        "medicine": "Appropriate insecticide depending on the specific pest identified.",
        "mixing_ratio_text": "Follow the product label recommendation.",
        "mixing_ratio_min_ml_per_l": None,
        "mixing_ratio_max_ml_per_l": None,
        "treatment": [
            "Inspect the field regularly to identify the specific pest.",
            "Spray the insecticide recommended for that pest.",
            "Remove heavily damaged leaves.",
        ],
        "prevention": [
            "Monitor the field regularly for early pest detection.",
            "Use pest-resistant varieties where available.",
        ],
        "important_note": "Exact medicine and dosage depend on the specific pest "
                           "species — identify the pest before treating. " + DEFAULT_IMPORTANT_NOTE,
    },
    "Potato___phytophthora": {
        "display_name": "Phytophthora",
        "status": "disease",
        "scientific_name": "Phytophthora spp.",
        "description": "An oomycete (water mold) infection causing irregular "
                        "brown lesions and rapid tissue decay in humid conditions. "
                        "Treated similarly to Late Blight.",
        "medicine": "Copper-based fungicide",
        "mixing_ratio_text": "2–3 ml per liter of water",
        "mixing_ratio_min_ml_per_l": 2.0,
        "mixing_ratio_max_ml_per_l": 3.0,
        "treatment": [
            "Spray fungicide immediately upon detection.",
            "Remove and destroy infected leaves.",
            "Improve field drainage.",
        ],
        "prevention": [
            "Avoid waterlogging in the field.",
            "Ensure good air circulation between plants.",
        ],
        "important_note": DEFAULT_IMPORTANT_NOTE,
    },
}
# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("potato_disease_app")

# --------------------------------------------------------------------------
# Flask app setup
# --------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# --------------------------------------------------------------------------
# Load the TFLite model + allocate tensors once per cold start
# --------------------------------------------------------------------------
interpreter = None
input_details = None
output_details = None
try:
    logger.info("Loading TFLite model from: %s", MODEL_PATH)
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("Model loaded successfully.")
except Exception as exc:  # noqa: BLE001
    logger.error("Failed to load model: %s", exc)
    interpreter = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes in memory and preprocess for the model."""
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the uploaded image. The file may be corrupted.")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def predict_disease(file_bytes: bytes):
    """
    Run the TFLite model on the uploaded image and return the detected
    disease with its full recommendation. Interpreter inference and
    preprocessing are unchanged — only the returned fields differ (no
    confidence/probability is exposed).
    """
    if interpreter is None:
        raise RuntimeError("Model is not loaded. Check server logs for details.")

    processed = preprocess_bytes(file_bytes)

    interpreter.set_tensor(input_details[0]["index"], processed)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]

    best_idx = int(np.argmax(predictions))
    best_key = CLASS_NAMES[best_idx]
    info = CLASS_INFO[best_key]

    dosage_text = calculate_dosage_text(
        info["mixing_ratio_min_ml_per_l"],
        info["mixing_ratio_max_ml_per_l"],
        info["medicine"],
    )

    return {
        "predicted_class": best_key,
        "display_name": info["display_name"],
        "status": info["status"],
        "scientific_name": info["scientific_name"],
        "description": info["description"],
        "medicine": info["medicine"],
        "mixing_ratio_text": info["mixing_ratio_text"],
        "dosage_text": dosage_text,
        "treatment": info["treatment"],
        "prevention": info["prevention"],
        "important_note": info["important_note"],
    }


def bytes_to_data_uri(file_bytes: bytes, mimetype: str) -> str:
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mimetype};base64,{encoded}"


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", model_loaded=interpreter is not None)


@app.route("/predict", methods=["POST"])
def predict():
    if interpreter is None:
        return jsonify({"success": False, "error": "Model is not loaded on the server."}), 500

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file was uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file was selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported file type. Please upload a PNG or JPG image.",
        }), 400

    try:
        file_bytes = file.read()

        result = predict_disease(file_bytes)
        result["success"] = True
        result["image_url"] = bytes_to_data_uri(file_bytes, file.mimetype or "image/jpeg")

        logger.info("Prediction complete: %s", result["display_name"])
        return jsonify(result), 200

    except ValueError as exc:
        logger.warning("Invalid image uploaded: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 400

    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return jsonify({"success": False, "error": f"Prediction failed: {exc}"}), 500


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"success": False, "error": "File is too large. Maximum size is 8 MB."}), 413


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"success": False, "error": "Route not found."}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"success": False, "error": "Internal server error."}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
