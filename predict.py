import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from cog import BasePredictor, Input, Path as CogPath
import segmentation_models_pytorch as smp
import yaml

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Load model
        self.model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights=None,
            in_channels=3,
            classes=4,
        )
        
        # Load trained weights
        checkpoint = torch.load("models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Class colors
        self.class_colors = {
            0: [0, 0, 0],        # background
            1: [255, 0, 0],      # building
            2: [0, 255, 0],      # woodland
            3: [0, 0, 255],      # water
        }
        
    def predict(
        self,
        image: CogPath = Input(description="Input satellite image"),
        overlay_alpha: float = Input(description="Overlay transparency", default=0.5, ge=0.0, le=1.0)
    ) -> CogPath:
        """Run segmentation"""
        
        # Load image
        img = cv2.imread(str(image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = img.copy()
        h, w = img.shape[:2]
        
        # Preprocess
        img_resized = cv2.resize(img, (512, 512))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(img_tensor)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Resize back
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Create colored mask
        colored_mask = np.zeros_like(original_img)
        for class_idx, color in self.class_colors.items():
            colored_mask[pred_mask == class_idx] = color
        
        # Blend
        result = cv2.addWeighted(original_img, 1 - overlay_alpha, colored_mask, overlay_alpha, 0)
        
        # Save
        output_path = Path("output.png")
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), result_bgr)
        
        return CogPath(output_path)