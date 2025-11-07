import torch
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import sys # <-- IMPORT SYS

# --- CRUCIAL IMPORT FIX ---
# We are forcing Python to look for files in this *exact* directory.
# This makes the import work regardless of how the app is run.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from legacy_models import get_model # <-- REMOVED THE DOT '.'
except ImportError:
    print("Error: Could not import 'get_model' from 'legacy_models.py'.")
        return None

class ManuscriptDamageDetector:
    def __init__(self, model_path, encoder, classes, device, threshold):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.model = self._load_model(model_path, encoder, classes)
        
        # Define the transformation from your notebook
        self.transform = A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(),
            ToTensorV2()
        ])
        print(f"✅ Model loaded successfully on device: {self.device}!")

    def _load_model(self, model_path, encoder, classes):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}. Please place 'best_model.pth' in the main project folder.")
            
        model = get_model(encoder_name=encoder, classes=classes)
        if model is None:
            raise ImportError("Model definition ('get_model') failed to load.")
            
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device).eval()
        return model

    def predict(self, image_path):
        """
        Runs the full prediction and post-processing pipeline from the notebook.
        """
        try:
            # --- 1. Load and Preprocess Image ---
            image = np.array(Image.open(image_path).convert("RGB"))
            aug = self.transform(image=image)
            input_tensor = aug["image"].unsqueeze(0).to(self.device)
            # Create grayscale image for masking (at original size)
            gray_img = np.mean(image / 255.0, axis=2)

            # --- 2. Model Prediction ---
            with torch.no_grad():
                pred = torch.sigmoid(self.model(input_tensor))
            
            # Get the damage probability map (channel 1)
            pred_map = pred.squeeze(0).cpu().numpy()[1]

            # --- 3. Binarization ---
            binary_pred = (pred_map > self.threshold).astype(np.uint8)

            # --- 4. Post-Processing (Noise Removal) ---
            binary_pred_denoised = cv2.medianBlur(binary_pred, 5)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_pred_denoised)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 400:  # remove small specks
                    binary_pred_denoised[labels == i] = 0

            # --- 5. Compute Page Mask (Ignore Margins) ---
            # We work at the 512x512 size, matching the notebook
            gray_resized = cv2.resize(gray_img, (512, 512))
            page_mask = gray_resized < 0.9  # Mask of the page text (not white margin)
            
            if np.sum(page_mask) == 0:
                print("Warning: Page mask is empty. Image might be all white.")
                page_mean = 0.9
                contrast = 0
            else:
                page_mean = np.mean(gray_resized[page_mask])
                contrast = np.std(gray_resized)

            brightness_factor = 0.75 if contrast > 0.15 else 0.9
            
            # Final damage candidates: must be in binary map AND darker than page mean
            damage_candidates = np.logical_and(
                binary_pred_denoised == 1, 
                gray_resized < (page_mean * brightness_factor)
            )

            # --- 6. Damage Percentage ---
            damage_pixels = np.sum(damage_candidates)
            total_pixels = np.sum(page_mask)
            damage_pct = (damage_pixels / total_pixels) * 100 if total_pixels > 0 else 0

            # --- 7. Create Visualization Arrays (RGB, 0-255) ---
            img_resized = cv2.resize(image, (512, 512))
            
            # Heatmap (0-1 probability map -> 0-255 heatmap)
            heatmap_normalized = (pred_map * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) # for PIL

            # Overlay (from notebook)
            overlay = img_resized.copy() / 255.0
            overlay[damage_candidates == 1] = [1.0, 0.0, 0.0]  # red highlight
            blended = np.clip(0.4 * overlay + 0.6 * (img_resized / 255.0), 0, 1)
            overlay_rgb = (blended * 255).astype(np.uint8)

            return {
                "damage_percentage": f"{damage_pct:.2f}%",
                "original_image_rgb": img_resized, # (512, 512, 3)
                "heatmap_image_rgb": heatmap_rgb,   # (512, 512, 3)
                "overlay_image_rgb": overlay_rgb    # (512, 512, 3)
            }
            
        except Exception as e:
            print(f"Error during prediction pipeline: {e}")
            raise