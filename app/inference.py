"""
Inference utilities for preprocessing, postprocessing, and output formatting.
Includes image colorization, area calculations, file handling, and urban risk analysis.
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

# =========================
# Class colors & class names
# =========================

# Example colors for: background, building, woodland, water, road
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

# =========================
# Colorization & Overlays
# =========================

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


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    class_colors: Dict[int, Tuple] = None
) -> np.ndarray:
    """
    Overlay segmentation mask on original image with transparency.

    This version keeps your existing optimization that automatically
    downsizes very large images to avoid memory issues.

    Args:
        image (np.ndarray): Original RGB image (H, W, 3).
        mask (np.ndarray): Segmentation mask (H, W).
        alpha (float): Transparency factor (0.0 to 1.0).
        class_colors (Dict[int, Tuple]): Mapping of class ID to RGB color.

    Returns:
        np.ndarray: Overlaid RGB image (H, W, 3).
    """
    # Automatically downscale large images to prevent memory errors
    MAX_SIZE = 2048
    h, w = image.shape[:2]
    if h > MAX_SIZE or w > MAX_SIZE:
        scale = min(MAX_SIZE / h, MAX_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    colored_mask = colorize_mask(mask, class_colors)
    image_f = image.astype(np.float32)
    colored_mask_f = colored_mask.astype(np.float32)
    overlaid = (image_f * (1 - alpha) + colored_mask_f * alpha).astype(np.uint8)
    return overlaid


def mask_to_base64_png(
    mask: np.ndarray,
    colorize: bool = True,
    class_colors: Dict[int, Tuple] = None
) -> str:
    """
    Convert segmentation mask (or RGB overlay) to a base64-encoded PNG data URL.

    Args:
        mask (np.ndarray): Segmentation mask (H, W) or RGB image (H, W, 3).
        colorize (bool): If True and mask is 2D, colorize using CLASS_COLORS.
        class_colors (Dict[int, Tuple]): Mapping of class ID to RGB color.

    Returns:
        str: Base64-encoded PNG string prefixed with "data:image/png;base64,".
    """
    try:
        if colorize:
            # Expect 2D label mask
            if mask.ndim != 2:
                raise ValueError("colorize=True expects a 2D label mask")
            rgb_mask = colorize_mask(mask, class_colors)
        else:
            # If already RGB, just use it as-is
            if mask.ndim == 3 and mask.shape[2] == 3:
                rgb_mask = mask
            else:
                # Convert to grayscale RGB (replicate channel)
                if mask.ndim == 2:
                    mask_uint8 = mask.astype(np.uint8)
                else:
                    # fallback: squeeze and then treat as grayscale
                    mask_uint8 = np.squeeze(mask).astype(np.uint8)
                rgb_mask = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=2)

        # Ensure proper shape and type
        if rgb_mask.ndim != 3 or rgb_mask.shape[2] != 3:
            logger.warning(f"Unexpected mask shape for PNG export: {rgb_mask.shape}")
            mask_uint8 = np.squeeze(rgb_mask).astype(np.uint8)
            rgb_mask = np.stack([mask_uint8, mask_uint8, mask_uint8], axis=2)

        pil_image = Image.fromarray(rgb_mask.astype(np.uint8), 'RGB')
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logger.error(f"Error converting mask to base64 PNG: {e}")
        raise

# =========================
# Class statistics & areas
# =========================

def compute_class_statistics(
    mask: np.ndarray,
    pixel_size_meters: Optional[float] = None,
    class_names: Dict[int, str] = None
) -> Dict:
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
    statistics: Dict[str, Any] = {
        "total_pixels": int(total_pixels),
        "per_class_pixels": {},
        "per_class_percentages": {}
    }

    unique_classes = np.unique(mask)

    for class_id in unique_classes:
        class_pixels = int((mask == class_id).sum())
        percentage = (class_pixels / total_pixels) * 100

        class_name = class_names.get(int(class_id), f"class_{int(class_id)}")
        statistics["per_class_pixels"][class_name] = class_pixels
        statistics["per_class_percentages"][class_name] = round(percentage, 2)

    # Compute areas if pixel size is provided
    if pixel_size_meters is not None:
        statistics["per_class_area_m2"] = {}
        statistics["per_class_area_km2"] = {}

        for class_id in unique_classes:
            class_pixels = int((mask == class_id).sum())
            area_m2 = float(class_pixels * (pixel_size_meters ** 2))
            area_km2 = float(area_m2 / 1e6)

            class_name = class_names.get(int(class_id), f"class_{int(class_id)}")
            statistics["per_class_area_m2"][class_name] = round(area_m2, 2)
            statistics["per_class_area_km2"][class_name] = round(area_km2, 6)

    return statistics

# =========================
# Temp file helpers
# =========================

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

# =========================
# Image & mask I/O
# =========================

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
            logger.warning(f"PIL failed: {pil_error}, trying opencv.")
            # Fallback to opencv
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError(f"Could not load image with either PIL or cv2")

            # Check if image has valid number of channels
            if image.ndim != 3 or image.shape[2] not in [3, 4]:
                logger.warning(f"Image has unexpected shape: {image.shape}, converting.")
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
            logger.warning(f"PIL failed: {pil_error}, trying cv2.")
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
        pil_image = Image.fromarray(mask.astype(np.uint8), 'L')
        pil_image.save(filepath, 'PNG')
        logger.info(f"Mask saved: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving mask to {filepath}: {e}")
        return False

# =========================
# Urban Risk Analysis helpers
# =========================

def mask_to_prob_arrays(
    pred_mask: np.ndarray,
    class_names_map: Dict[int, str]
) -> Dict[str, np.ndarray]:
    """
    Convert a prediction mask (labels or probabilities) into per-class float arrays.

    Supports:
    - 2D integer mask (H, W) with class IDs
    - 3D probabilities (C, H, W) or (H, W, C), where C == len(class_names_map)

    Returns:
        Dict[str, np.ndarray]: Mapping from class name to float array in [0,1].
    """
    arr = np.asarray(pred_mask)
    out: Dict[str, np.ndarray] = {}

    num_classes = len(class_names_map)

    # 2D label mask
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        for cid, cname in class_names_map.items():
            out[cname] = (arr == cid).astype(np.float32)
        return out

    # 3D probabilities
    if arr.ndim == 3:
        h, w = arr.shape[:2]

        # (C, H, W)
        if arr.shape[0] == num_classes:
            for idx, (cid, cname) in enumerate(class_names_map.items()):
                out[cname] = arr[idx].astype(np.float32)
            return out

        # (H, W, C)
        if arr.shape[2] == num_classes:
            for idx, (cid, cname) in enumerate(class_names_map.items()):
                out[cname] = arr[:, :, idx].astype(np.float32)
            return out

    # Fallback: treat as 2D label mask via argmax
    logger.warning("mask_to_prob_arrays: unexpected shape, falling back to argmax")
    if arr.ndim >= 3:
        labels = np.argmax(arr, axis=-1)
    else:
        labels = arr
    for cid, cname in class_names_map.items():
        out[cname] = (labels == cid).astype(np.float32)
    return out


def to_binary(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert array to boolean mask using a threshold."""
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return arr
    return arr > threshold


def erode_binary(binary_mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Erode a boolean mask."""
    kernel = np.ones((ksize, ksize), np.uint8)
    eroded = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=iterations)
    return eroded.astype(bool)


def dilate_binary(binary_mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Dilate a boolean mask."""
    kernel = np.ones((ksize, ksize), np.uint8)
    dilated = cv2.dilate(binary_mask.astype(np.uint8), kernel, iterations=iterations)
    return dilated.astype(bool)


def label_connected_components(binary_mask: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Label connected components in a boolean mask.

    Returns:
        num_labels (int): Number of labels (including background as 0).
        labels (np.ndarray): Labeled mask (H, W).
    """
    binary_u8 = binary_mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary_u8)
    return num_labels, labels


def compute_buildings_near_water(
    building_mask: np.ndarray,
    water_mask: np.ndarray,
    buffer_pixels: int = 3
) -> Tuple[int, int, float]:
    """
    Compute how many building components lie near water, given a pixel buffer.

    Args:
        building_mask (np.ndarray): Float or boolean mask of buildings.
        water_mask (np.ndarray): Float or boolean mask of water.
        buffer_pixels (int): Buffer radius in pixels.

    Returns:
        (total_buildings, buildings_near_water, pct_near_water)
    """
    bld = to_binary(building_mask)
    wtr = to_binary(water_mask)

    if bld.sum() == 0:
        return 0, 0, 0.0

    num_labels, labels = label_connected_components(bld)
    total_buildings = max(0, int(num_labels) - 1)  # label 0 is background

    if total_buildings == 0:
        return 0, 0, 0.0

    if wtr.sum() == 0:
        return total_buildings, 0, 0.0

    if buffer_pixels <= 0:
        buffer_pixels = 1

    water_buffer = dilate_binary(wtr, ksize=3, iterations=buffer_pixels)

    near_count = 0
    for label_id in range(1, num_labels):
        comp_mask = (labels == label_id)
        if np.any(comp_mask & water_buffer):
            near_count += 1

    pct = (near_count / total_buildings) * 100.0
    return int(total_buildings), int(near_count), float(pct)


def compute_risk_map(
    pred_mask: np.ndarray,
    class_names_map: Dict[int, str] = None,
    tile_size: int = 256,
    pixel_size_meters: Optional[float] = None,
    weights: Dict[str, float] = None,
    buffer_meters: float = 30.0
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """
    Compute a simple urban risk map based on land-cover classes.

    Args:
      pred_mask: 2D integer mask or probability map.
      class_names_map: {class_id -> class_name} mapping.
      tile_size: tile side in pixels for aggregation.
      pixel_size_meters: ground pixel size in meters (if known).
      weights: dict of weights for different classes.
      buffer_meters: distance from water in meters to count building as "near water".

    Returns:
      risk_map: float32 array (H, W) in [0,1] representing per-pixel risk.
      risk_tiles: list[dict] where each tile dict contains at least:
        {"tile_id", "x", "y", "w", "h", "building_ratio", "risk_score", "risk_level"}
      risk_summary: dict with aggregate fields such as:
        {"mean_risk", "high_risk_tile_pct", "buildings_near_water_pct",
         "total_buildings", "buildings_near_water",
         "roads_coverage_pct", "roads_missing"}
      risk_flags: list of human-readable string flags (badges).
    """
    if class_names_map is None:
        class_names_map = CLASS_NAMES

    if weights is None:
        weights = {
            "water": 0.5,
            "building": 0.3,
            "woodland": 0.2,
            "road_penalty": 0.25
        }

    # Convert to per-class float arrays
    arrs = mask_to_prob_arrays(pred_mask, class_names_map)
    h, w = pred_mask.shape[:2]

    water = arrs.get("water", np.zeros((h, w), dtype=np.float32))
    building = arrs.get("building", np.zeros((h, w), dtype=np.float32))
    woodland = arrs.get("woodland", np.zeros((h, w), dtype=np.float32))
    road = arrs.get("road", np.zeros((h, w), dtype=np.float32))

    # Basic pixel-wise raw score
    raw = (
        weights["water"] * water +
        weights["building"] * building -
        weights["woodland"] * woodland +
        weights["road_penalty"] * (1.0 - road)
    )
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
            road_block = road[y:min(y + tile_size, h), x:min(x + tile_size, w)]
            if block.size == 0:
                continue

            mean_risk = float(np.nanmean(block))
            bld_ratio = float(np.sum(bld_block) / block.size)
            road_ratio = float(np.sum(road_block) / block.size)

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
                "road_ratio": round(road_ratio, 6),
                "risk_score": round(mean_risk, 6),
                "risk_level": level
            })
            tile_id += 1
            total_tiles += 1

    high_risk_tile_pct = (high_count / total_tiles) * 100.0 if total_tiles > 0 else 0.0
    mean_risk = float(np.nanmean(risk)) if risk.size > 0 else 0.0

    # Compute building <-> water proximity using dilation buffer (pixel-based)
    if pixel_size_meters and pixel_size_meters > 0:
        buffer_pixels = max(1, int(round(buffer_meters / pixel_size_meters)))
    else:
        # default buffer in pixels if no georef provided
        buffer_pixels = max(3, int(round(tile_size * 0.02)))  # e.g. ~5 px for 256 tiles

    total_buildings, buildings_near_water, buildings_near_water_pct = compute_buildings_near_water(
        building, water, buffer_pixels=buffer_pixels
    )

    roads_coverage_pct = float(road.mean() * 100.0) if road.size > 0 else 0.0
    roads_missing = roads_coverage_pct < 5.0  # heuristic: <5% roads in entire scene

    risk_summary: Dict[str, Any] = {
        "mean_risk": round(mean_risk, 4),
        "high_risk_tile_pct": round(high_risk_tile_pct, 2),
        "total_buildings": int(total_buildings),
        "buildings_near_water": int(buildings_near_water),
        "buildings_near_water_pct": round(buildings_near_water_pct, 2),
        "roads_coverage_pct": round(roads_coverage_pct, 2),
        "roads_missing": bool(roads_missing),
    }

    # Flags for UI badges
    risk_flags: List[str] = []
    if high_risk_tile_pct > 30:
        risk_flags.append(f"High-risk hotspots across {high_risk_tile_pct:.1f}% of tiles")
    if buildings_near_water_pct > 20:
        risk_flags.append(f"{buildings_near_water_pct:.1f}% of buildings near water (flood-sensitive)")
    if roads_missing:
        risk_flags.append("Road coverage appears sparse – potential access issues")

    # Always include at least one flag so UI looks populated
    if not risk_flags:
        risk_flags.append("Overall risk appears moderate")

    return risk.astype(np.float32), tiles, risk_summary, risk_flags
