"""
Inference utilities for preprocessing, postprocessing, and output formatting.
Includes image colorization, area calculations, and file handling.
"""

import numpy as np
import cv2
import base64
import io
from PIL import Image
from typing import Dict, Tuple, Optional
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

# TODO: Customize color mapping based on your land-cover classes
# These are example colors for: background, building, woodland, water, road
CLASS_COLORS = {
    0: (0, 0, 0),           # background - black
    1: (200, 0, 0),         # building - red
    2: (34, 139, 34),       # woodland - forest green
    3: (0, 149, 218),       # water - blue
    4: (128, 128, 128)      # road - gray
}

CLASS_NAMES = {
    0: 'background',
    1: 'building',
    2: 'woodland',
    3: 'water',
    4: 'road'
}


def colorize_mask(mask: np.ndarray, class_colors: Dict[int, Tuple] = None) -> np.ndarray:
    """
    Convert single-channel mask to RGB colorized visualization.
    
    Args:
        mask (np.ndarray): Single-channel mask (H, W) with class IDs.
        class_colors (Dict[int, Tuple]): Mapping of class ID to RGB color.
    
    Returns:
        np.ndarray: RGB colorized image (H, W, 3).
    
    TODO: Allow custom color schemes via configuration.
    """
    if class_colors is None:
        class_colors = CLASS_COLORS
    
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in class_colors.items():
        colored[mask == class_id] = color
    
    return colored


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5,
                          class_colors: Dict[int, Tuple] = None) -> np.ndarray:
    """
    Overlay segmentation mask on original image with transparency.
    
    Args:
        image (np.ndarray): Original RGB image (H, W, 3).
        mask (np.ndarray): Segmentation mask (H, W).
        alpha (float): Transparency factor (0.0 to 1.0).
        class_colors (Dict[int, Tuple]): Mapping of class ID to RGB color.
    
    Returns:
        np.ndarray: Overlaid RGB image (H, W, 3).
    """
    colored_mask = colorize_mask(mask, class_colors)
    # Use numpy blend instead of cv2.addWeighted for better compatibility
    overlaid = (image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    return overlaid


def mask_to_base64_png(mask: np.ndarray, colorize: bool = True,
                      class_colors: Dict[int, Tuple] = None) -> str:
    """
    Convert segmentation mask to base64-encoded PNG string for embedding in responses.
    
    Args:
        mask (np.ndarray): Segmentation mask (H, W).
        colorize (bool): Whether to colorize the mask.
        class_colors (Dict[int, Tuple]): Mapping of class ID to RGB color.
    
    Returns:
        str: Base64-encoded PNG string.
    """
    try:
        if colorize:
            rgb_mask = colorize_mask(mask, class_colors)
        else:
            # Convert to RGB with grayscale (replicate on all 3 channels)
            if mask.ndim == 2:
                mask_uint8 = mask.astype(np.uint8)
                rgb_mask = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=2)
            else:
                rgb_mask = mask
        
        # Ensure proper shape and type
        if rgb_mask.ndim != 3 or rgb_mask.shape[2] != 3:
            logger.warning(f"Unexpected mask shape: {rgb_mask.shape}, converting to grayscale RGB")
            if rgb_mask.ndim == 2:
                mask_uint8 = rgb_mask.astype(np.uint8)
                rgb_mask = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=2)
        
        # Convert to PIL Image and then to PNG bytes
        pil_image = Image.fromarray(rgb_mask.astype(np.uint8), 'RGB')
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting mask to base64 PNG: {e}")
        raise


def compute_class_statistics(mask: np.ndarray, pixel_size_meters: Optional[float] = None,
                            class_names: Dict[int, str] = None) -> Dict:
    """
    Compute per-class pixel counts and optional area estimates.
    
    Args:
        mask (np.ndarray): Segmentation mask (H, W).
        pixel_size_meters (Optional[float]): Ground pixel size in meters for area computation.
        class_names (Dict[int, str]): Mapping of class ID to class name.
    
    Returns:
        Dict: Dictionary with per-class statistics (pixel counts, percentages, areas if pixel_size provided).
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    total_pixels = mask.size
    statistics = {
        "total_pixels": int(total_pixels),
        "per_class_pixels": {},
        "per_class_percentages": {}
    }
    
    unique_classes = np.unique(mask)
    
    for class_id in unique_classes:
        class_pixels = int((mask == class_id).sum())
        percentage = (class_pixels / total_pixels) * 100
        
        class_name = class_names.get(class_id, f"class_{class_id}")
        statistics["per_class_pixels"][class_name] = class_pixels
        statistics["per_class_percentages"][class_name] = round(percentage, 2)
    
    # Compute areas if pixel size is provided
    if pixel_size_meters is not None:
        statistics["per_class_area_m2"] = {}
        statistics["per_class_area_km2"] = {}
        
        for class_id in unique_classes:
            class_pixels = (mask == class_id).sum()
            area_m2 = float(class_pixels * (pixel_size_meters ** 2))
            area_km2 = float(area_m2 / 1e6)
            
            class_name = class_names.get(class_id, f"class_{class_id}")
            statistics["per_class_area_m2"][class_name] = round(area_m2, 2)
            statistics["per_class_area_km2"][class_name] = round(area_km2, 6)
    
    return statistics


def save_temporary_file(file_bytes: bytes, suffix: str = '.png') -> str:
    """
    Save uploaded file to temporary directory.
    
    Args:
        file_bytes (bytes): File content.
        suffix (str): File extension (default: '.png').
    
    Returns:
        str: Path to temporary file.
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file.write(file_bytes)
    temp_file.close()
    logger.info(f"Temporary file saved: {temp_file.name}")
    return temp_file.name


def cleanup_temporary_file(filepath: str) -> bool:
    """
    Remove temporary file.
    
    Args:
        filepath (str): Path to file to delete.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Temporary file deleted: {filepath}")
            return True
    except Exception as e:
        logger.error(f"Error deleting temporary file {filepath}: {e}")
    return False


def load_image_as_rgb(filepath: str) -> np.ndarray:
    """
    Load image from file and convert to RGB.
    Handles various image formats including TIFF, PNG, JPG.
    
    Args:
        filepath (str): Path to image file.
    
    Returns:
        np.ndarray: RGB image (H, W, 3).
    """
    try:
        # Try PIL first as it's more reliable on headless systems
        from PIL import Image as PILImage
        try:
            pil_img = PILImage.open(filepath)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            image = np.array(pil_img)
            logger.info(f"Image loaded via PIL: {image.shape}")
            return image
        except Exception as pil_error:
            logger.warning(f"PIL failed: {pil_error}, trying opencv...")
            # Fallback to opencv
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError(f"Could not load image with either PIL or cv2")
            
            # Check if image has valid number of channels
            if image.ndim != 3 or image.shape[2] not in [3, 4]:
                logger.warning(f"Image has unexpected shape: {image.shape}, converting...")
                # Convert grayscale or RGBA to RGB
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Standard BGR to RGB conversion
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            logger.info(f"Image loaded via cv2: {image.shape}")
            return image
    except Exception as e:
        logger.error(f"Error loading image from {filepath}: {e}")
        raise ValueError(f"Cannot read image from {filepath}: {e}")


def load_mask_as_uint8(filepath: str) -> np.ndarray:
    """
    Load mask from file as grayscale uint8.
    
    Args:
        filepath (str): Path to mask file.
    
    Returns:
        np.ndarray: Grayscale mask (H, W) with class IDs.
    """
    try:
        # Try PIL first (more reliable on headless systems)
        from PIL import Image as PILImage
        try:
            pil_img = PILImage.open(filepath)
            if pil_img.mode != 'L':
                pil_img = pil_img.convert('L')
            mask = np.array(pil_img, dtype=np.uint8)
            logger.info(f"Mask loaded via PIL: {mask.shape}, dtype: {mask.dtype}")
            return mask
        except Exception as pil_error:
            logger.warning(f"PIL failed: {pil_error}, trying cv2...")
            # Fallback to opencv
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Cannot read mask from {filepath}")
            logger.info(f"Mask loaded via cv2: {mask.shape}, dtype: {mask.dtype}")
            return mask
    except Exception as e:
        raise ValueError(f"Cannot read mask from {filepath}: {e}")


def save_image_as_png(image: np.ndarray, filepath: str) -> bool:
    """
    Save RGB image as PNG file.
    
    Args:
        image (np.ndarray): RGB image (H, W, 3).
        filepath (str): Output file path.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Use PIL to save (more reliable than cv2)
        pil_image = Image.fromarray(image.astype(np.uint8), 'RGB')
        pil_image.save(filepath, 'PNG')
        logger.info(f"Image saved: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving image to {filepath}: {e}")
        return False


def save_mask_as_png(mask: np.ndarray, filepath: str) -> bool:
    """
    Save segmentation mask as grayscale PNG file.
    
    Args:
        mask (np.ndarray): Segmentation mask (H, W).
        filepath (str): Output file path.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Use PIL to save (more reliable than cv2)
        pil_image = Image.fromarray(mask.astype(np.uint8), 'L')
        pil_image.save(filepath, 'PNG')
        logger.info(f"Mask saved: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving mask to {filepath}: {e}")
        return False
