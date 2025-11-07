import os
import gc
import torch
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Import your model logic
from model.manuscript_model import ManuscriptDamageDetector

# ---------------- Configuration ----------------
CONFIG = {
    "model_path": "best_model.pth",  # Must exist in root folder
    "encoder": "resnet34",
    "classes": 2,
    "device": "cpu",  # ✅ Force CPU mode (Render has no GPU)
    "threshold": 0.85
}

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.secret_key = "supersecretkey_replace_me"
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # ✅ Limit upload to 8 MB

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------- Lazy Model Load ----------------
manuscript_detector = None

def get_model_instance():
    """Load model only once (lazy loading)."""
    global manuscript_detector
    if manuscript_detector is None:
        torch.set_grad_enabled(False)
        manuscript_detector = ManuscriptDamageDetector(
            model_path=CONFIG["model_path"],
            encoder=CONFIG["encoder"],
            classes=CONFIG["classes"],
            device=CONFIG["device"],
            threshold=CONFIG["threshold"]
        )
        print("✅ Model loaded successfully in CPU mode.")
    return manuscript_detector


# ---------------- Utility Functions ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def encode_image_to_base64(image_array_rgb):
    """Convert NumPy RGB array (0–255) to base64 string."""
    img = Image.fromarray(image_array_rgb.astype("uint8"), "RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", optimize=True, quality=80)  # smaller images
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    """Main prediction route — runs damage detection."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        detector = get_model_instance()
        results = detector.predict(filepath)

        # Convert to base64 for the frontend
        original_b64 = encode_image_to_base64(results["original_image_rgb"])
        heatmap_b64 = encode_image_to_base64(results["heatmap_image_rgb"])
        overlay_b64 = encode_image_to_base64(results["overlay_image_rgb"])

        response = {
            "original_image": original_b64,
            "heatmap_image": heatmap_b64,
            "overlay_image": overlay_b64,
            "damage_percentage": results.get("damage_percentage", "N/A")
        }

        # ✅ Cleanup memory aggressively
        del results
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return jsonify(response)

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    finally:
        # ✅ Delete uploaded file after use
        if os.path.exists(filepath):
            os.remove(filepath)


# ---------------- Graceful Shutdown ----------------
@app.teardown_appcontext
def cleanup(exception=None):
    """Release model and clear memory when Flask shuts down."""
    global manuscript_detector
    if manuscript_detector:
        del manuscript_detector
        manuscript_detector = None
    gc.collect()
    print("🧹 Cleaned up memory and model resources.")


# ---------------- App Runner ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # ✅ Render assigns this dynamically
    print(f"🚀 Flask app running on port {port} (Render auto-assigned)")
    app.run(host="0.0.0.0", port=port)
