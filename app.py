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
    "model_path": "best_model.pth",   # ensure this file exists at root
    "encoder": "resnet34",
    "classes": 2,
    "device": "cpu",                  # ✅ Force CPU mode (Render has no GPU)
    "threshold": 0.85
}

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.secret_key = "supersecretkey_replace_me"
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # ✅ Reduced max upload (8 MB)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------- Lazy Model Load ----------------
manuscript_detector = None

def get_model_instance():
    """Load model only once (lazy loading)."""
    global manuscript_detector
    if manuscript_detector is None:
        torch.set_grad_enabled(False)  # ✅ Disable autograd globally
        manuscript_detector = ManuscriptDamageDetector(
            model_path=CONFIG["model_path"],
            encoder=CONFIG["encoder"],
            classes=CONFIG["classes"],
            device=CONFIG["device"],
            threshold=CONFIG["threshold"]
        )
        print("✅ Model loaded successfully (CPU mode).")
    return manuscript_detector


# ---------------- Utility Functions ----------------
def allowed_file(filename):
    """Check allowed file extensions."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def encode_image_to_base64(image_array_rgb):
    """Convert NumPy RGB array to base64 string."""
    img = Image.fromarray(image_array_rgb.astype("uint8"), "RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", optimize=True, quality=85)  # ✅ Lower quality = smaller memory
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------- Routes ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        # ✅ Lazy load model only when needed
        detector = get_model_instance()

        # ✅ Run prediction
        results = detector.predict(filepath)

        # ✅ Convert outputs for frontend
        original_b64 = encode_image_to_base64(results["original_image_rgb"])
        heatmap_b64 = encode_image_to_base64(results["heatmap_image_rgb"])
        overlay_b64 = encode_image_to_base64(results["overlay_image_rgb"])

        # ✅ Explicit cleanup to release memory
        del results
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return jsonify({
            "original_image": original_b64,
            "heatmap_image": heatmap_b64,
            "overlay_image": overlay_b64,
            "damage_percentage": results.get("damage_percentage", "N/A")
        })

    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return jsonify({"error": f"Error processing image: {e}"}), 500
    finally:
        # ✅ Cleanup uploaded file after processing
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify({"error": "Unexpected error"}), 500


# ---------------- Graceful Shutdown ----------------
@app.teardown_appcontext
def cleanup(exception=None):
    """Release model and clear memory on shutdown."""
    global manuscript_detector
    if manuscript_detector:
        del manuscript_detector
        manuscript_detector = None
    gc.collect()
    print("🧹 Cleaned up model and memory.")

# ---------------- App Runner ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
