import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
import torch

# Import your model logic
from model.manuscript_model import ManuscriptDamageDetector

# --- Configuration ---
CONFIG = {
    "model_path": "best_model.pth",   # make sure this file exists in root
    "encoder": "resnet34",
    "classes": 2,
    "device": "cpu",                  # ✅ Force CPU mode for Render (saves ~400MB)
    "threshold": 0.85
}

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = 'supersecretkey_replace_me'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Lazy Load Model ---
manuscript_detector = None

def get_model_instance():
    """Load ManuscriptDamageDetector only once (lazy init to save memory)."""
    global manuscript_detector
    if manuscript_detector is None:
        try:
            manuscript_detector = ManuscriptDamageDetector(
                model_path=CONFIG["model_path"],
                encoder=CONFIG["encoder"],
                classes=CONFIG["classes"],
                device=CONFIG["device"],
                threshold=CONFIG["threshold"]
            )
            print("✅ Model loaded successfully (CPU mode).")
        except Exception as e:
            print(f"❌ Model initialization failed: {e}")
            manuscript_detector = None
    return manuscript_detector

# --- Utility Functions ---
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def encode_image_to_base64(image_array_rgb):
    """Convert NumPy RGB array (0–255) to base64 string."""
    img = Image.fromarray(image_array_rgb.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        detector = get_model_instance()
        if detector is None:
            return jsonify({'error': 'Model could not be loaded. Check logs.'}), 500

        try:
            # Run damage detection
            results = detector.predict(filepath)

            # Convert to base64 for frontend display
            original_img_base64 = encode_image_to_base64(results["original_image_rgb"])
            heatmap_img_base64 = encode_image_to_base64(results["heatmap_image_rgb"])
            overlay_img_base64 = encode_image_to_base64(results["overlay_image_rgb"])

            return jsonify({
                'original_image': original_img_base64,
                'heatmap_image': heatmap_img_base64,
                'overlay_image': overlay_img_base64,
                'damage_percentage': results["damage_percentage"]
            })

        except Exception as e:
            print(f"❌ Error processing image {filename}: {e}")
            return jsonify({'error': f'Error processing image: {e}'}), 500

    return jsonify({'error': 'Invalid file type'}), 400

# --- App Runner ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
