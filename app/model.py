"""
Model loading and inference module for land-cover segmentation.
Handles PyTorch model loading with safety features and tiled inference for large images.
"""

import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from patchify import patchify, unpatchify
import math
import logging
from typing import Tuple

# Allowlist the Unet class for safe loading with PyTorch 2.6+
try:
    from torch.serialization import add_safe_globals
    from segmentation_models_pytorch.decoders.unet.model import Unet
    add_safe_globals([Unet])
except Exception as e:
    logging.warning(f"Could not add safe globals for Unet: {e}")

logger = logging.getLogger(__name__)


class SegmentationModel:
    """
    Wrapper for segmentation model with inference capabilities.
    Supports tiled inference for large images.
    """
    
    def __init__(self, checkpoint_path: str, encoder: str = 'efficientnet-b0', 
                 encoder_weights: str = 'imagenet', num_classes: int = 5, 
                 device: str = None):
        """
        Initialize the segmentation model.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint.
            encoder (str): Encoder backbone name.
            encoder_weights (str): Pre-trained weights for encoder.
            num_classes (int): Number of output classes.
            device (str): Device to load model on ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        
        logger.info(f"Initializing model on device: {self.device}")
        self.model = self._load_model(checkpoint_path)
        self.preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        logger.info("Model loaded successfully")

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """
        Load model checkpoint safely.
        
        Args:
            checkpoint_path (str): Path to checkpoint file.
        
        Returns:
            torch.nn.Module: Loaded model in eval mode.
        """
        try:
            # Try loading with weights_only=False for PyTorch 2.6+ compatibility
            model = torch.load(checkpoint_path, map_location=torch.device(self.device), 
                             weights_only=False)
            model.eval()
            logger.info(f"Model loaded from {checkpoint_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {checkpoint_path}: {e}")
            raise

    def predict(self, image: np.ndarray, patch_size: int = 512, 
                step_size: int = None) -> np.ndarray:
        """
        Performs inference on a single image with optional tiling for large images.
        
        Args:
            image (np.ndarray): Input image (H, W, 3) in RGB format.
            patch_size (int): Size of patches for tiling (default: 512).
            step_size (int): Step size for patch extraction (default: patch_size for non-overlapping).
        
        Returns:
            np.ndarray: Predicted segmentation mask (H, W) with class IDs.
        
        TODO: Adjust patch_size and step_size based on GPU memory constraints for production.
        TODO: Consider smoothing overlapping regions for better boundary predictions.
        """
        # Use non-overlapping patches to avoid unpatchify issues
        step_size = patch_size  # Non-overlapping patches
        
        original_height, original_width = image.shape[:2]
        
        # Pad image to be divisible by patch_size
        pad_height = (math.ceil(original_height / patch_size) * patch_size) - original_height
        pad_width = (math.ceil(original_width / patch_size) * patch_size) - original_width
        padded_shape = ((0, pad_height), (0, pad_width), (0, 0))
        image_padded = np.pad(image, padded_shape, mode='reflect')
        
        logger.info(f"Original image shape: {image.shape}, Padded: {image_padded.shape}")
        
        # Patchify with non-overlapping step
        patches = patchify(image_padded, (patch_size, patch_size, 3), step=step_size)
        
        # Handle edge case where image is smaller than patch size
        if patches.ndim == 5:
            patches = patches[:, :, 0, :, :, :]
        
        mask_patches = np.zeros(patches.shape[:-1], dtype=np.uint8)
        
        logger.info(f"Processing {patches.shape[0]} x {patches.shape[1]} patches")
        
        with torch.no_grad():
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    # Preprocess patch (output shape depends on preprocessing_fn)
                    img_patch = self.preprocessing_fn(patches[i, j, :, :, :])
                    
                    # Handle batch dimension if preprocessing_fn added it
                    if img_patch.ndim == 4:  # (1, H, W, C) or (1, C, H, W)
                        img_patch = img_patch.squeeze(0)
                    
                    # Ensure img_patch is in (C, H, W) format
                    if img_patch.ndim == 3 and img_patch.shape[2] == 3:
                        # Shape is (H, W, C) - need to transpose
                        img_patch = img_patch.transpose(2, 0, 1).astype('float32')
                    elif img_patch.ndim == 3 and img_patch.shape[0] == 3:
                        # Already (C, H, W)
                        img_patch = img_patch.astype('float32')
                    else:
                        logger.warning(f"Unexpected patch shape: {img_patch.shape}")
                        if img_patch.ndim == 3:
                            img_patch = img_patch.transpose(2, 0, 1).astype('float32')
                        else:
                            img_patch = img_patch.astype('float32')
                    
                    x_tensor = torch.from_numpy(img_patch).to(self.device).unsqueeze(0)
                    
                    # Model prediction
                    try:
                        pred_logits = self.model.predict(x_tensor)
                        pred_mask = pred_logits.squeeze().cpu().numpy()
                        
                        # Apply argmax for multi-class segmentation
                        if pred_mask.ndim == 3:  # (C, H, W)
                            pred_mask = pred_mask.argmax(axis=0)
                        else:  # Already (H, W)
                            pred_mask = pred_mask
                        
                        mask_patches[i, j, :, :] = pred_mask.astype(np.uint8)
                    except Exception as e:
                        logger.error(f"Error predicting patch ({i}, {j}): {e}")
                        mask_patches[i, j, :, :] = 0  # Default to background class
        
        # Unpatchify with matching padded dimensions
        try:
            pred_mask_padded = unpatchify(mask_patches, (image_padded.shape[0], image_padded.shape[1]))
        except Exception as e:
            logger.error(f"Error in unpatchify: {e}. Mask patches shape: {mask_patches.shape}, Padded shape: {image_padded.shape[:2]}")
            # Fallback: stack patches manually
            pred_mask_padded = np.zeros((image_padded.shape[0], image_padded.shape[1]), dtype=np.uint8)
            for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                    y_start = i * patch_size
                    x_start = j * patch_size
                    pred_mask_padded[y_start:y_start+patch_size, x_start:x_start+patch_size] = mask_patches[i, j, :, :]
        
        # Remove padding
        final_pred_mask = pred_mask_padded[:original_height, :original_width]
        
        logger.info(f"Inference complete. Output shape: {final_pred_mask.shape}")
        return final_pred_mask
