import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import sys
import importlib

# --- CRUCIAL IMPORT FIX ---
# Dynamically import get_model from model/legacy_models.py, no hardcoded path
try:
    legacy = importlib.import_module("model.legacy_models")
    get_model = getattr(legacy, "get_model")
except Exception as e:
    print(f"❌ Failed to import get_model from model/legacy_models.py: {e}")
    get_model = None


class ManuscriptDamageDetector:
    def __init__(self, model_path, encoder, classes, device, threshold):
        """
        Main class for damage detection inference.
        Loads the trained model, applies preprocessing, prediction, and visualization.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model = self._load_model(model_path, encoder, classes)

        # Preprocessing transformations
        self.transform = A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(),
            ToTensorV2()
        ])

        print(f"✅ Model initialized successfully on device: {self.device}")

    def _load_model(self, model_path, encoder, classes):
        """
        Loads the PyTorch model with weights.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                f"Please upload 'best_model.pth' to the project root."
            )

        if get_model is None:
            raise ImportError("Could not import get_model() from model/legacy_models.py.")

        model = get_model(encoder_name=encoder, classes=classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device).eval()
        return model

    def predict(self, image_path):
        """
        Runs the full prediction pipeline and returns results (damage %, images).
        """
        try:
            # --- Step 1: Load and preprocess ---
            image = np.array(Image.open(image_path).convert("RGB"))
            aug = self.transform(image=image)
            input_tensor = aug["image"].unsqueeze(0).to(self.device)
            gray_img = np.mean(image / 255.0, axis=2)

            # --- Step 2: Predict ---
            with torch.no_grad():
                pred = torch.sigmoid(self.model(input_tensor))
            pred_map = pred.squeeze(0).cpu().numpy()[1]

            # --- Step 3: Threshold & denoise ---
            binary_pred = (pred_map > self.threshold).astype(np.uint8)
            binary_pred = cv2.medianBlur(binary_pred, 5)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_pred)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 400:
                    binary_pred[labels == i] = 0

            # --- Step 4: Page mask and damage detection ---
            gray_resized = cv2.resize(gray_img, (512, 512))
            page_mask = gray_resized < 0.9
            if np.sum(page_mask) == 0:
                print("⚠️ Warning: Page mask is empty. Possibly all-white image.")
                page_mean, contrast = 0.9, 0
            else:
                page_mean = np.mean(gray_resized[page_mask])
                contrast = np.std(gray_resized)

            brightness_factor = 0.75 if contrast > 0.15 else 0.9
            damage_candidates = np.logical_and(
                binary_pred == 1, gray_resized < (page_mean * brightness_factor)
            )

            # --- Step 5: Damage percentage ---
            damage_pixels = np.sum(damage_candidates)
            total_pixels = np.sum(page_mask)
            damage_pct = (damage_pixels / total_pixels) * 100 if total_pixels > 0 else 0

            # --- Step 6: Visualizations ---
            img_resized = cv2.resize(image, (512, 512))

            # Heatmap
            heatmap = (pred_map * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Overlay
            overlay = img_resized.copy() / 255.0
            overlay[damage_candidates == 1] = [1.0, 0.0, 0.0]
            blended = np.clip(0.4 * overlay + 0.6 * (img_resized / 255.0), 0, 1)
            overlay_rgb = (blended * 255).astype(np.uint8)

            return {
                "damage_percentage": f"{damage_pct:.2f}%",
                "original_image_rgb": img_resized,
                "heatmap_image_rgb": heatmap_rgb,
                "overlay_image_rgb": overlay_rgb
            }

        except Exception as e:
            print(f"❌ Error during prediction pipeline: {e}")
            raise
