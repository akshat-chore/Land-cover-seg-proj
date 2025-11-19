"""
Inference utilities for preprocessing, postprocessing, and output formatting.
Includes image colorization, area calculations, and file handling.
"""

import numpy as np
import cv2
import base64
import io
from PIL import Image
from typing import Dict, Tuple, Optional, List, Any
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


# ---------------------------
# Urban Risk Analysis helpers
# ---------------------------

def mask_to_prob_arrays(label_mask: np.ndarray, class_names_map: Optional[Dict[int, str]] = None) -> Dict[str, np.ndarray]:
    """
    Convert integer label mask (HxW) -> dict of per-class float arrays (0/1).
    class_names_map: {0:'background',1:'building',...}
    Returns: {"building": arr, "water": arr, ...} arrays dtype float32 in [0,1]
    """
    if class_names_map is None:
        class_names_map = CLASS_NAMES

    h, w = label_mask.shape[:2]
    arrs: Dict[str, np.ndarray] = {}
    for cid, name in class_names_map.items():
        try:
            arrs[name] = (label_mask == int(cid)).astype(np.float32)
        except Exception:
            arrs[name] = np.zeros((h, w), dtype=np.float32)
    return arrs


# ---------- Robust connected-component + dilation fallbacks ----------
from collections import deque

def label_connected_components(bin_arr: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Label connected components in a binary 2D uint8 array.
    Tries scipy.ndimage.label, otherwise uses a BFS flood-fill fallback.
    Returns (num_labels_plus_one, labels_array) where labels start at 1 and 0 is background.
    """
    # Try scipy first (recommended, fast)
    try:
        import scipy.ndimage as ndi
        labels, num = ndi.label(bin_arr, structure=np.ones((3,3), dtype=np.int32))
        # ndi.label returns labels starting at 1 and num = number of features
        return int(num) + 1, labels.astype(np.int32)  # +1 to mimic cv2.connectedComponents behaviour
    except Exception:
        pass

    # Fallback: BFS flood fill labeling (pure Python + numpy). Labels start at 1.
    h, w = bin_arr.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current_label = 1
    visited = np.zeros_like(bin_arr, dtype=np.uint8)

    # 8-connectivity neighbors
    neighbor_offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for i in range(h):
        for j in range(w):
            if bin_arr[i, j] and not visited[i, j]:
                # BFS
                q = deque()
                q.append((i,j))
                visited[i, j] = 1
                labels[i, j] = current_label
                while q:
                    y,x = q.popleft()
                    for dy,dx in neighbor_offsets:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if bin_arr[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = 1
                                labels[ny, nx] = current_label
                                q.append((ny, nx))
                current_label += 1

    # mimic cv2.connectedComponents return: num_labels = current_label (including background as label 0 not counted)
    num_labels = current_label  # this equals (#components + 1)
    return num_labels, labels


def dilate_binary(bin_arr: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Dilate a binary array by radius_px pixels.
    Tries scipy.ndimage.binary_dilation, otherwise uses a simple iterative neighbor expansion.
    """
    if radius_px <= 0:
        return bin_arr.copy()
    # Try scipy
    try:
        import scipy.ndimage as ndi
        struct = ndi.generate_binary_structure(2, 2)
        # create disk-like structuring element approximately by successive dilations
        out = bin_arr.copy().astype(bool)
        out = ndi.binary_dilation(out, structure=struct, iterations=radius_px)
        return out.astype(np.uint8)
    except Exception:
        pass

    # Fallback: iterative 8-neighbor dilation radius_px times (slower but works)
    out = bin_arr.copy().astype(np.uint8)
    h, w = out.shape
    for _ in range(radius_px):
        # create shifted OR of neighbors
        up    = np.pad(out[:-1,:], ((1,0),(0,0)), mode='constant')
        down  = np.pad(out[1:,:], ((0,1),(0,0)), mode='constant')
        left  = np.pad(out[:,:-1], ((0,0),(1,0)), mode='constant')
        right = np.pad(out[:,1:], ((0,0),(0,1)), mode='constant')
        ul = np.pad(out[:-1,:-1], ((1,0),(1,0)), mode='constant')
        ur = np.pad(out[:-1,1:], ((1,0),(0,1)), mode='constant')
        dl = np.pad(out[1:,:-1], ((0,1),(1,0)), mode='constant')
        dr = np.pad(out[1:,1:], ((0,1),(0,1)), mode='constant')
        out = np.clip(out | up | down | left | right | ul | ur | dl | dr, 0, 1).astype(np.uint8)
    return out


def compute_buildings_near_water(building_arr: np.ndarray, water_arr: np.ndarray, buffer_pixels: int = 10) -> Tuple[int, int, float]:
    """
    Robust computation of connected building components near water, with fallbacks
    if OpenCV or SciPy are missing.
    Returns: (total_building_components, num_near_water, pct_near_water)
    """
    # Binarize arrays
    bld = (building_arr > 0.5).astype(np.uint8)
    wtr = (water_arr > 0.5).astype(np.uint8)

    # Quick exits
    if np.sum(bld) == 0:
        return 0, 0, 0.0
    if np.sum(wtr) == 0:
        # no water => no buildings near water
        # but still count components
        num_labels, labels = label_connected_components(bld)
        total = max(0, int(num_labels) - 1)
        return total, 0, 0.0

    # Label building components
    num_labels, labels = label_connected_components(bld)
    total = max(0, int(num_labels) - 1)
    if total == 0:
        return 0, 0, 0.0

    # Create water buffer (dilation)
    if buffer_pixels <= 0:
        buffer_pixels = 1
    water_buffer = dilate_binary(wtr, buffer_pixels)

    # For each labeled component check overlap with buffer
    near_count = 0
    # labels values start at 1 ... num_labels-1
    for label_id in range(1, num_labels):
        comp_mask = (labels == label_id)
        if np.any(comp_mask & water_buffer.astype(bool)):
            near_count += 1

    pct = (near_count / total) * 100.0 if total > 0 else 0.0
    return int(total), int(near_count), float(pct)

def compute_risk_map(
    pred_mask: np.ndarray,
    class_names_map: Optional[Dict[int, str]] = None,
    tile_size: int = 256,
    weights: Optional[Dict[str, float]] = None,
    pixel_size_meters: Optional[float] = None,
    buffer_meters: float = 50.0
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """
    Compute per-pixel risk map and aggregated tile risk scores.

    Args:
      pred_mask: HxW integer mask (class ids)
      class_names_map: mapping of ids to names (like CLASS_NAMES). If None, uses CLASS_NAMES.
      tile_size: tile size in pixels for aggregation
      weights: dict with keys 'water','building','woodland','road_penalty' (defaults used if None)
      pixel_size_meters: if provided, used to convert buffer_meters -> buffer_pixels
      buffer_meters: distance in meters to consider building near water (if pixel_size_meters known)

    Returns:
      risk_map: HxW float array in [0,1]
      tiles: list of dicts {"tile_id", "x","y","w","h","building_ratio","risk_score","risk_level"}
      risk_summary: {"mean_risk", "high_risk_tile_pct","buildings_near_water_pct","roads_missing", ...}
      risk_flags: list of strings (badges)
    """
    if class_names_map is None:
        class_names_map = CLASS_NAMES

    if weights is None:
        weights = {"water": 0.5, "building": 0.3, "woodland": 0.2, "road_penalty": 0.25}

    # Convert to per-class float arrays
    arrs = mask_to_prob_arrays(pred_mask, class_names_map)
    h, w = pred_mask.shape[:2]
    water = arrs.get("water", np.zeros((h, w), dtype=np.float32))
    building = arrs.get("building", np.zeros((h, w), dtype=np.float32))
    woodland = arrs.get("woodland", np.zeros((h, w), dtype=np.float32))
    road = arrs.get("road", np.zeros((h, w), dtype=np.float32))

    # Basic pixel-wise raw score
    raw = weights["water"] * water + weights["building"] * building - weights["woodland"] * woodland + weights["road_penalty"] * (1.0 - road)
    risk = np.clip(raw, 0.0, 1.0)

    # Aggregate into tiles
    tiles: List[Dict[str, Any]] = []
    tile_id = 0
    high_count = 0
    total_tiles = 0
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            block = risk[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            bld_block = building[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            # avoid empty blocks
            if block.size == 0:
                continue
            mean_risk = float(np.nanmean(block))
            bld_ratio = float(np.sum(bld_block) / block.size)
            if mean_risk >= 0.75:
                level = "high"
                high_count += 1
            elif mean_risk >= 0.5:
                level = "medium"
            else:
                level = "low"
            tiles.append({
                "tile_id": tile_id,
                "x": int(x),
                "y": int(y),
                "w": int(min(tile_size, w - x)),
                "h": int(min(tile_size, h - y)),
                "building_ratio": round(bld_ratio, 6),
                "risk_score": round(mean_risk, 6),
                "risk_level": level
            })
            tile_id += 1
            total_tiles += 1

    high_risk_tile_pct = (high_count / total_tiles) * 100.0 if total_tiles > 0 else 0.0
    mean_risk = float(np.nanmean(risk)) if risk.size > 0 else 0.0

    # Compute building <-> water proximity using dilation buffer (pixel-based)
    # Determine buffer_pixels
    if pixel_size_meters and pixel_size_meters > 0:
        buffer_pixels = max(1, int(round(buffer_meters / pixel_size_meters)))
    else:
        # default buffer in pixels if no georef provided
        buffer_pixels = max(3, int(round(tile_size * 0.02)))  # e.g., ~5 px for 256 tiles

    total_buildings, buildings_near_water, buildings_near_water_pct = compute_buildings_near_water(
        building, water, buffer_pixels
    )

    # Road missing flag
    total_pixels = h * w
    road_pixel_count = int(np.sum(road > 0.5))
    roads_missing = (road_pixel_count / total_pixels) < 0.005  # threshold 0.5%

    # Build summary and flags
    risk_summary: Dict[str, Any] = {
        "mean_risk": round(mean_risk, 6),
        "high_risk_tile_pct": round(high_risk_tile_pct, 2),
        "total_tiles": total_tiles,
        "total_building_components": total_buildings,
        "buildings_near_water": int(buildings_near_water),
        "buildings_near_water_pct": round(buildings_near_water_pct, 2),
        "road_pixel_count": int(road_pixel_count)
    }

    risk_flags: List[str] = []
    if roads_missing:
        risk_flags.append("roads_missing")
    if high_risk_tile_pct >= 10.0:  # if >=10% tiles high risk
        risk_flags.append("high_flood_exposure")
    if (np.sum(building) / total_pixels) >= 0.20:
        risk_flags.append("overbuilt_area")

    return risk, tiles, risk_summary, risk_flags
