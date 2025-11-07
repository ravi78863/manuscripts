import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
import torch

# Import model logic
from model.manuscript_model import ManuscriptDamageDetector

# --- Configuration ---
CONFIG = {
    "model_path": "best_model.pth",
    "encoder": "resnet34",
    "classes": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "threshold": 0.85
}
# --- End Configuration ---

app = Flask(__name__)
app.secret_key = 'supersecretkey_replace_me'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Lazy-load model to save memory
manuscript_detector = None

def get_detector():
    """Load the model only when first needed."""
    global manuscript_detector
    if manuscript_detector is None:
        print("⚙️ Loading model now...")
        manuscript_detector = ManuscriptDamageDetector(
            model_path=CONFIG["model_path"],
            encoder=CONFIG["encoder"],
            classes=CONFIG["classes"],
            device=CONFIG["device"],
            threshold=CONFIG["threshold"]
        )
        print("✅ Model loaded successfully!")
    return manuscript_detector

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def encode_image_to_base64(image_array_rgb):
    """Convert NumPy RGB array (0-255) to base64 string."""
    img = Image.fromarray(image_array_rgb.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

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

        try:
            detector = get_detector()  # Load model only when needed
            results = detector.predict(filepath)

            # Encode results to base64
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
            print(f"Error processing image {filename}: {e}")
            return jsonify({'error': f'Error processing image: {e}'}), 500

    return jsonify({'error': 'File type not allowed'}), 400

@app.route("/health")
def health():
    """Simple route for Render health check."""
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
