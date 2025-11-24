"""
FastAPI server for land-cover semantic segmentation.
Provides endpoints for prediction, evaluation, report generation,
urban risk mapping, flood simulation, infrastructure equity (deficit),
and disaster response planning.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import time
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import base64
import tempfile

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import cv2
import numpy as np
import rasterio
from rasterio.warp import transform_bounds

# Import custom modules
from app.model import SegmentationModel
from app.metrics import compute_all_metrics
from app.gemini_client import generate_report_sync
from app.inference import (
    load_image_as_rgb,
    load_mask_as_uint8,
    colorize_mask,
    overlay_mask_on_image,
    mask_to_base64_png,
    compute_class_statistics,
    save_temporary_file,
    cleanup_temporary_file,
    compute_risk_map,  # risk map + tiles
)
from app.flood_engine import run_simulation_pipeline  # flood simulation

# ======================= Logging Setup =======================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# ======================= Configuration =======================

MODEL_CHECKPOINT = os.getenv(
    "MODEL_CHECKPOINT",
    "models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth",
)
DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "5"))
ENCODER = os.getenv("ENCODER", "efficientnet-b0")
ENCODER_WEIGHTS = os.getenv("ENCODER_WEIGHTS", "imagenet")
SERVER_API_KEY = os.getenv("SERVER_API_KEY", None)  # Optional API key for server
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)  # Gemini API key for report generation
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Class names mapping
CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "building",
    2: "woodland",
    3: "water",
    4: "road",
}

# ======================= Global State =======================

model: Optional[SegmentationModel] = None
model_loaded: bool = False
device_info: str = DEVICE

# ======================= Model Initialization =======================


def load_model_on_startup() -> None:
    """Load model at startup with proper error handling."""
    global model, model_loaded, device_info
    print(f"[STARTUP] Loading model from {MODEL_CHECKPOINT}")
    logger.info(f"Loading model from {MODEL_CHECKPOINT}")
    logger.info(f"Model checkpoint path: {MODEL_CHECKPOINT}")
    logger.info(f"Device: {DEVICE}, Encoder: {ENCODER}, Num Classes: {NUM_CLASSES}")

    try:
        import sys

        logger.info(f"Python: {sys.version}")
        logger.info(f"PyTorch available: {torch.cuda.is_available()}")

        print("[STARTUP] Initializing SegmentationModel.")
        model = SegmentationModel(
            checkpoint_path=MODEL_CHECKPOINT,
            encoder=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            num_classes=NUM_CLASSES,
            device=DEVICE,
        )
        model_loaded = True
        device_info = model.device
        print(f"[STARTUP] ✓ Model loaded successfully on device: {device_info}")
        logger.info(f"Model loaded successfully on device: {device_info}")
    except Exception as e:
        import traceback

        error_msg = f"Failed to load model: {e}\n{traceback.format_exc()}"
        print(f"[STARTUP] ✗ {error_msg}")
        logger.error(error_msg)
        model_loaded = False
        raise


# ======================= Startup & Shutdown =======================


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Load model synchronously
    print("[LIFESPAN] Starting application lifespan")
    try:
        load_model_on_startup()
    except Exception as e:
        print(f"[LIFESPAN] Error during model loading: {e}")
        logger.error(f"Error during model loading in lifespan: {e}")

    yield

    # Shutdown
    print("[LIFESPAN] Shutting down application")
    logger.info("Shutting down application")


# ======================= FastAPI App =======================

app = FastAPI(
    title="Land-Cover Segmentation API",
    description=(
        "API for land-cover semantic segmentation with evaluation metrics, "
        "urban risk mapping, flood simulation, infrastructure equity analysis, "
        "and Gemini-powered reports"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================= Dependencies =======================


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """Optional API key verification."""
    if SERVER_API_KEY:
        if x_api_key != SERVER_API_KEY:
            raise HTTPException(
                status_code=403, detail="Invalid or missing X-API-Key header"
            )
    return True


def ensure_model_loaded() -> SegmentationModel:
    """Ensure model is loaded before processing."""
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server startup logs.",
        )
    return model


# ======================= Pydantic Models =======================


class PredictResponse(BaseModel):
    """Response model for /predict endpoint."""

    success: bool
    inference_time_ms: float
    image_shape: tuple
    mask_shape: tuple
    mask_base64: str = Field(
        ..., description="Base64-encoded colorized mask PNG"
    )
    overlay_base64: str = Field(
        ..., description="Base64-encoded overlay PNG"
    )
    class_statistics: Dict[str, Any]
    unique_classes: List[int] = Field(
        ..., description="Classes found in prediction"
    )

    # Risk
    risk_summary: Optional[Dict[str, Any]] = Field(
        None, description="Summary metrics for risk"
    )
    risk_tiles: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tile-level risk scores"
    )
    risk_flags: Optional[List[str]] = Field(
        None, description="Flags like roads_missing, high_flood_exposure"
    )
    timestamp: str

    # Geo fields
    geo_bounds: Optional[List[float]] = Field(
        None, description="[lat_min, lon_min, lat_max, lon_max]"
    )
    geo_center: Optional[List[float]] = Field(
        None, description="[lat, lon] center point"
    )
    has_geo: bool = Field(
        False, description="Whether image has geographic coordinates"
    )

    # NEW: Infrastructure equity / deficit
    equity_summary: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Summary of infrastructure deficit / equity per tile "
            "(mean_deficit_score, total_tiles, urgent/important/adequate counts, "
            "optional cost totals)"
        ),
    )
    equity_tiles: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "Per-tile infrastructure deficit metrics: deficit_score (0–10), "
            "priority_level, building_density_pct, road_coverage_pct, "
            "green_space_pct, optional cost_estimate, lat/lon if available"
        ),
    )

    # Flood simulation outputs
    flood_metrics: Optional[Dict[str, Any]] = Field(
        None,
        description="Flood simulation metrics (impact_ratio, flooded_buildings, status)",
    )
    flood_gif_base64: Optional[str] = Field(
        None, description="Base64-encoded GIF for flood simulation animation"
    )

    # Disaster response deployment plan
    response_plan: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Recommended deployment of response teams per tile (when & where)",
    )


class EvaluateResponse(BaseModel):
    """Response model for /evaluate endpoint."""

    success: bool
    inference_time_ms: float
    metrics: Dict[str, Any]
    class_statistics: Dict[str, Any]
    confusion_matrix: List[List[int]]  # 2D array
    unique_pred_classes: List[int]
    unique_gt_classes: List[int]
    timestamp: str


class ReportRequest(BaseModel):
    """Request model for /report endpoint."""

    metrics_json: Dict[str, Any] = Field(..., description="Metrics from evaluation")
    segmentation_summary: Dict[str, Any] = Field(
        ..., description="Summary of segmentation results"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Optional context (region, date, scenario)"
    )


class ReportResponse(BaseModel):
    """Response model for /report endpoint."""

    success: bool
    status: str
    report: Optional[Dict[str, Any]] = Field(
        None, description="Generated report with insights"
    )
    raw_prompt: Optional[str] = Field(
        None, description="Exact prompt sent to Gemini"
    )
    raw_response: Optional[str] = Field(
        None, description="Raw Gemini response"
    )
    error: Optional[str] = Field(
        None, description="Error message if generation failed"
    )
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""

    status: str
    model_loaded: bool
    device: str
    api_key_present: bool
    timestamp: str


# ======================= Helper: GeoTIFF Bounds =======================


def extract_geotiff_bounds(image_path: str) -> Dict[str, Any]:
    """
    Extract geographic bounds from a GeoTIFF file.
    Returns lat/lon bounds and a function to transform pixel coords to lat/lon.
    """
    try:
        with rasterio.open(image_path) as src:
            bounds = src.bounds  # (left, bottom, right, top)
            original_crs = src.crs

            # Transform to WGS84 (lat/lon) for web mapping
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                original_crs,
                "EPSG:4326",
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            )

            # Image dimensions
            height, width = src.height, src.width

            def pixel_to_latlon(x: float, y: float) -> Tuple[float, float]:
                """Convert pixel coordinates to lat/lon."""
                lon = lon_min + (x / width) * (lon_max - lon_min)
                # y is inverted
                lat = lat_max - (y / height) * (lat_max - lat_min)
                return float(lat), float(lon)

            return {
                "bounds": [float(lat_min), float(lon_min), float(lat_max), float(lon_max)],
                "center": [
                    float((lat_min + lat_max) / 2),
                    float((lon_min + lon_max) / 2),
                ],
                "pixel_to_latlon": pixel_to_latlon,
                "has_geo": True,
            }
    except Exception as e:
        logger.warning(f"Could not extract geo info: {e}")
        return {"has_geo": False}


# ======================= Helper: Infrastructure Equity / Deficit =======================


def compute_infrastructure_equity(
    pred_mask: np.ndarray,
    risk_tiles: List[Dict[str, Any]],
    pixel_size_meters: Optional[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Compute infrastructure deficit scores and simple cost estimates per tile.

    Uses segmentation classes:
      - building
      - road
      - woodland (as green space)

    For each tile:
      - building_density_pct
      - road_coverage_pct
      - green_space_pct
      - deficit_score (0–10)
      - priority_level: urgent / important / adequate
      - optional cost_estimate (relative, based on area & deficit)
    """
    if not risk_tiles:
        return [], {}

    # Resolve class IDs from CLASS_NAMES mapping (fallback to defaults)
    building_id = next(
        (k for k, v in CLASS_NAMES.items() if v == "building"), 1
    )
    road_id = next(
        (k for k, v in CLASS_NAMES.items() if v == "road"), 4
    )
    green_id = next(
        (k for k, v in CLASS_NAMES.items() if v in ("woodland", "green", "park")),
        2,
    )

    h, w = pred_mask.shape[:2]

    equity_tiles: List[Dict[str, Any]] = []

    # Targets for "adequate" infrastructure (in % of tile pixels)
    target_building = 20.0
    target_road = 10.0
    target_green = 15.0

    # Base cost per m^2 (arbitrary scale factor)
    base_cost_per_m2 = 50.0

    total_deficit = 0.0
    urgent_count = 0
    important_count = 0
    adequate_count = 0
    total_cost = 0.0

    for tile in risk_tiles:
        x = int(tile.get("x", 0))
        y = int(tile.get("y", 0))
        w_tile = int(tile.get("w", 0))
        h_tile = int(tile.get("h", 0))

        if w_tile <= 0 or h_tile <= 0:
            continue

        # Clip to image bounds
        x2 = min(x + w_tile, w)
        y2 = min(y + h_tile, h)
        x1 = max(x, 0)
        y1 = max(y, 0)

        if x1 >= x2 or y1 >= y2:
            continue

        tile_mask = pred_mask[y1:y2, x1:x2]
        tile_pixels = tile_mask.size
        if tile_pixels == 0:
            continue

        building_pixels = int((tile_mask == building_id).sum())
        road_pixels = int((tile_mask == road_id).sum())
        green_pixels = int((tile_mask == green_id).sum())

        building_pct = 100.0 * building_pixels / tile_pixels
        road_pct = 100.0 * road_pixels / tile_pixels
        green_pct = 100.0 * green_pixels / tile_pixels

        # Compute deficits vs target (only when below target)
        building_def = max(0.0, target_building - building_pct)
        road_def = max(0.0, target_road - road_pct)
        green_def = max(0.0, target_green - green_pct)

        # Simple combined deficit score, scaled to 0–10
        combined_def = building_def + road_def + green_def  # max could be ~45
        deficit_score = max(0.0, min(10.0, combined_def / 4.5))  # 0–10

        # Priority based on deficit
        if deficit_score >= 7.0:
            priority = "urgent"
            urgent_count += 1
        elif deficit_score >= 4.0:
            priority = "important"
            important_count += 1
        else:
            priority = "adequate"
            adequate_count += 1

        # Area-based cost estimate (if pixel_size_meters is known)
        if pixel_size_meters and pixel_size_meters > 0:
            tile_area_m2 = (w_tile * pixel_size_meters) * (h_tile * pixel_size_meters)
        else:
            # Fallback: use pixel count as proxy
            tile_area_m2 = float(tile_pixels)

        cost_estimate = (
            base_cost_per_m2 * tile_area_m2 * (deficit_score / 10.0)
        )

        total_deficit += deficit_score
        total_cost += cost_estimate

        equity_tile = {
            "tile_id": tile.get("tile_id"),
            "x": x,
            "y": y,
            "w": w_tile,
            "h": h_tile,
            "building_density_pct": building_pct,
            "road_coverage_pct": road_pct,
            "green_space_pct": green_pct,
            "deficit_score": deficit_score,
            "priority_level": priority,
            "cost_estimate": cost_estimate,
        }

        # Preserve lat/lon if already added in risk_tiles
        if "lat" in tile and "lon" in tile:
            equity_tile["lat"] = tile["lat"]
            equity_tile["lon"] = tile["lon"]

        equity_tiles.append(equity_tile)

    n_tiles = len(equity_tiles)
    equity_summary: Dict[str, Any] = {
        "total_tiles": n_tiles,
        "mean_deficit_score": (total_deficit / n_tiles) if n_tiles > 0 else 0.0,
        "urgent_tiles": urgent_count,
        "important_tiles": important_count,
        "adequate_tiles": adequate_count,
        "total_estimated_cost": total_cost,
        "mean_cost_per_tile": (total_cost / n_tiles) if n_tiles > 0 else 0.0,
    }

    return equity_tiles, equity_summary


# ======================= Endpoints =======================


@app.get("/", tags=["Server"])
async def root():
    """Root endpoint - redirects to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Server"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns server status, model availability, and device information.
    """
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        device=device_info,
        api_key_present=GEMINI_API_KEY is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(
    image: UploadFile = File(
        ...,
        description="Input satellite/aerial image (PNG, JPG, TIFF)",
    ),
    metadata: Optional[str] = Form(
        None,
        description="Optional JSON metadata (pixel_size_meters, etc.)",
    ),
    confidence_threshold: float = Form(
        0.5,
        description=(
            "Confidence threshold (not used in semantic segmentation but kept for compatibility)"
        ),
    ),
    _: bool = Depends(verify_api_key),
    segmentation_model: SegmentationModel = Depends(ensure_model_loaded),
) -> PredictResponse:
    """
    Run inference on a single satellite image.

    Returns:
    - Segmentation mask (base64 PNG)
    - Colorized overlay (base64 PNG)
    - Per-class statistics
    - Urban risk analysis (tile-level risk, summary, flags)
    - Infrastructure equity (deficit) metrics per tile
    - Flood simulation metrics + GIF
    - Disaster response deployment plan
    """
    try:
        start_time = time.time()

        # Parse metadata if provided
        meta_dict: Dict[str, Any] = {}
        pixel_size_meters: Optional[float] = None
        if metadata:
            try:
                meta_dict = json.loads(metadata)
                if meta_dict.get("pixel_size_meters") is not None:
                    pixel_size_meters = float(meta_dict.get("pixel_size_meters"))
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON provided")
            except Exception:
                logger.warning("Invalid pixel_size_meters value in metadata")

        # Save and load image
        image_bytes = await image.read()
        image_path = save_temporary_file(image_bytes, suffix=".png")
        geo_info: Dict[str, Any] = {"has_geo": False}
        try:
            geo_info = extract_geotiff_bounds(image_path)
            image_rgb = load_image_as_rgb(image_path)
        finally:
            # For /predict we keep file until after risk & flood (we need geo, but
            # geo_info is already extracted; we clean up later in risk block).
            pass

        # Run inference
        logger.info(f"Running inference on image shape {image_rgb.shape}")
        pred_mask = segmentation_model.predict(image_rgb, patch_size=512)

        # ----------------- SANITIZE pred_mask to a 2D integer label mask -----------------
        pred_mask = np.asarray(pred_mask)

        # If model returned a batch dim (1, ...)
        if pred_mask.ndim == 4 and pred_mask.shape[0] == 1:
            pred_mask = pred_mask[0]

        # If model returned (C, H, W) probability map -> argmax over channel
        if pred_mask.ndim == 3:
            if pred_mask.shape[0] == NUM_CLASSES:
                # shape (C, H, W)
                pred_mask = np.argmax(pred_mask, axis=0).astype(np.uint8)
            elif pred_mask.shape[2] == NUM_CLASSES:
                # shape (H, W, C)
                pred_mask = np.argmax(pred_mask, axis=2).astype(np.uint8)
            else:
                # maybe (1, H, W) or (H, W, 1)
                if pred_mask.shape[0] == 1:
                    pred_mask = pred_mask[0]
                elif pred_mask.shape[2] == 1:
                    pred_mask = pred_mask[:, :, 0]
                else:
                    # fallback: take channel-wise argmax
                    try:
                        pred_mask = np.argmax(pred_mask, axis=0).astype(np.uint8)
                    except Exception:
                        pred_mask = np.squeeze(pred_mask)

        # If mask is float, round/clip to nearest int
        if pred_mask.dtype != np.uint8 and np.issubdtype(pred_mask.dtype, np.floating):
            try:
                if (
                    pred_mask.max() <= 1.0
                    and pred_mask.min() >= 0.0
                    and pred_mask.ndim == 2
                ):
                    pred_mask = (pred_mask > 0.5).astype(np.uint8)
                else:
                    pred_mask = np.round(pred_mask).astype(np.uint8)
            except Exception:
                pred_mask = np.round(pred_mask).astype(np.uint8)

        # Ensure it's 2D
        if pred_mask.ndim != 2:
            pred_mask = np.squeeze(pred_mask)
            if pred_mask.ndim != 2:
                try:
                    pred_mask = np.argmax(pred_mask, axis=-1).astype(np.uint8)
                except Exception:
                    pred_mask = pred_mask.astype(np.uint8)
        # ----------------- END SANITIZE -----------------

        # Colorize mask and create overlay
        colored_mask = colorize_mask(pred_mask)
        overlay = overlay_mask_on_image(image_rgb, pred_mask, alpha=0.5)

        # Compute class statistics
        class_stats = compute_class_statistics(
            pred_mask, pixel_size_meters, CLASS_NAMES
        )

        # 🧮 Compute Urban Risk Map and Tile Scores
        try:
            risk_map, risk_tiles, risk_summary, risk_flags = compute_risk_map(
                pred_mask,
                class_names_map=CLASS_NAMES,
                tile_size=256,
                pixel_size_meters=pixel_size_meters,
            )
            if geo_info.get("has_geo") and "pixel_to_latlon" in geo_info:
                for tile in risk_tiles:
                    center_x = tile["x"] + tile["w"] / 2
                    center_y = tile["y"] + tile["h"] / 2
                    lat, lon = geo_info["pixel_to_latlon"](center_x, center_y)
                    tile["lat"] = lat
                    tile["lon"] = lon

        except Exception as re:
            logger.exception("Error computing risk map, skipping risk outputs: %s", re)
            risk_map, risk_tiles, risk_summary, risk_flags = None, [], {}, []
        finally:
            # Cleanup temp file now that geo is extracted
            cleanup_temporary_file(image_path)

        # NEW: Infrastructure equity / deficit computation (per-tile)
        equity_tiles: List[Dict[str, Any]] = []
        equity_summary: Dict[str, Any] = {}
        try:
            equity_tiles, equity_summary = compute_infrastructure_equity(
                pred_mask=pred_mask,
                risk_tiles=risk_tiles or [],
                pixel_size_meters=pixel_size_meters,
            )
        except Exception as ee:
            logger.exception(
                "Error computing infrastructure equity, skipping equity outputs: %s",
                ee,
            )
            equity_tiles, equity_summary = [], {}

        # Prepare response plan container (so it's always defined)
        response_plan: List[Dict[str, Any]] = []

        # 🌊 Flood Simulation based on predicted mask (buildings vs water)
        flood_metrics: Optional[Dict[str, Any]] = None
        flood_gif_base64: Optional[str] = None
        tmp_gif_path: Optional[str] = None
        try:
            fd, tmp_gif_path = tempfile.mkstemp(suffix=".gif")
            os.close(fd)

            # Downscale mask for simulation speed (e.g. max dimension 512px)
            h, w = pred_mask.shape
            max_dim = 512
            scale = 1.0
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
            if scale != 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                simulation_mask = cv2.resize(
                    pred_mask,
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                simulation_mask = pred_mask

            flood_result = run_simulation_pipeline(simulation_mask, tmp_gif_path)

            # Map flood simulation back to tiles using building_flood_mask
            building_flood_mask = flood_result.get("building_flood_mask")

            if building_flood_mask is not None and risk_tiles:
                # Resize building_flood_mask back to prediction size if needed
                if building_flood_mask.shape != pred_mask.shape:
                    building_flood_mask_resized = cv2.resize(
                        building_flood_mask.astype(np.uint8),
                        (pred_mask.shape[1], pred_mask.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    building_flood_mask_resized = building_flood_mask

                for tile in risk_tiles:
                    x = int(tile.get("x", 0))
                    y = int(tile.get("y", 0))
                    w_tile = int(tile.get("w", 0))
                    h_tile = int(tile.get("h", 0))

                    if w_tile <= 0 or h_tile <= 0:
                        tile["flooded_building_ratio"] = 0.0
                        tile["flooded_buildings"] = 0
                        tile["total_buildings"] = 0
                        continue

                    tile_pred = pred_mask[y : y + h_tile, x : x + w_tile]
                    tile_buildings = tile_pred == 1  # building class
                    total_b = int(tile_buildings.sum())
                    if total_b == 0:
                        tile["flooded_building_ratio"] = 0.0
                        tile["flooded_buildings"] = 0
                        tile["total_buildings"] = 0
                        continue

                    tile_flood_mask = building_flood_mask_resized[
                        y : y + h_tile, x : x + w_tile
                    ]
                    tile_flooded = tile_buildings & tile_flood_mask
                    flooded_b = int(tile_flooded.sum())
                    flood_ratio = flooded_b / total_b

                    tile["flooded_building_ratio"] = float(flood_ratio)
                    tile["flooded_buildings"] = flooded_b
                    tile["total_buildings"] = total_b

                    # Upgrade risk_level if simulation shows heavy building flooding
                    old_level = tile.get("risk_level", "low")
                    if flood_ratio >= 0.6:
                        tile["risk_level"] = "high"
                    elif flood_ratio >= 0.3 and old_level == "low":
                        tile["risk_level"] = "medium"

            # Keep only the key metrics in JSON response
            flood_metrics = {
                "impact_ratio": flood_result.get("impact_ratio"),
                "total_buildings": flood_result.get("total_buildings"),
                "flooded_buildings": flood_result.get("flooded_buildings"),
                "status": flood_result.get("status"),
            }

            # Attach GIF as base64 string
            with open(tmp_gif_path, "rb") as f:
                flood_gif_base64 = base64.b64encode(f.read()).decode("utf-8")

            # === DISASTER RESPONSE PLAN (WHEN & WHERE TEAMS GO) ===
            try:
                overall_status = (flood_metrics or {}).get("status", "UNKNOWN")
                for tile in risk_tiles or []:
                    tile_id = tile.get("tile_id")
                    risk_level = tile.get("risk_level", "low")
                    building_ratio = tile.get("building_ratio", 0.0)
                    flooded_ratio = tile.get("flooded_building_ratio", 0.0)
                    lat = tile.get("lat")
                    lon = tile.get("lon")

                    # Ignore empty/no-building tiles
                    if building_ratio < 0.05 or tile.get("total_buildings", 0) == 0:
                        continue

                    # Priority rules using BOTH static risk + simulated flooding
                    if flooded_ratio >= 0.6:
                        priority = "urgent"
                        time_window = "0–1 hours"
                        action = (
                            "Immediate rescue & evacuation (heavy building flooding)"
                        )
                    elif flooded_ratio >= 0.3:
                        priority = "important"
                        time_window = "1–3 hours"
                        action = (
                            "Evacuation support & medical team (moderate flooding)"
                        )
                    else:
                        # fall back to static risk_level when flood_ratio is low
                        if risk_level == "high":
                            priority = "urgent"
                            time_window = "0–1 hours"
                            action = "Immediate rescue & evacuation (high static risk)"
                        elif risk_level == "medium":
                            priority = "important"
                            time_window = "1–3 hours"
                            action = "Evacuation support & field medical team"
                        else:
                            priority = "monitor"
                            time_window = "3–6 hours"
                            action = "Supply preparation & monitoring"

                    # Boost text if global flood status is CRITICAL
                    if overall_status.startswith("CRITICAL") and priority in (
                        "urgent",
                        "important",
                    ):
                        action += " (CRITICAL FLOOD ALERT)"

                    response_plan.append(
                        {
                            "tile_id": tile_id,
                            "lat": lat,
                            "lon": lon,
                            "risk_level": risk_level,
                            "priority": priority,
                            "recommended_window": time_window,
                            "recommended_action": action,
                            "eta_minutes": 0
                            if priority == "urgent"
                            else 60
                            if priority == "important"
                            else 180,
                            "building_ratio": building_ratio,
                            "flooded_building_ratio": flooded_ratio,
                            "flooded_buildings": tile.get("flooded_buildings", 0),
                            "total_buildings": tile.get("total_buildings", 0),
                        }
                    )

                # Sort by priority (urgent > important > monitor),
                # then by flooded_building_ratio desc
                priority_rank = {"urgent": 0, "important": 1, "monitor": 2}
                response_plan.sort(
                    key=lambda x: (
                        priority_rank.get(x["priority"], 3),
                        -x.get("flooded_building_ratio", 0.0),
                    )
                )

            except Exception as e:
                logger.exception("Error generating response plan: %s", e)
                response_plan = []

            # If status is critical, push a flag into risk_flags
            if flood_metrics.get("status", "").startswith("CRITICAL"):
                if risk_flags is None:
                    risk_flags = []
                risk_flags.append("flood_critical")

        except Exception as fe:
            logger.exception(
                "Error during flood simulation, skipping flood outputs: %s", fe
            )
        finally:
            if tmp_gif_path and os.path.exists(tmp_gif_path):
                try:
                    os.remove(tmp_gif_path)
                except OSError:
                    pass

        # Convert to base64
        mask_base64 = mask_to_base64_png(pred_mask, colorize=True)
        overlay_base64 = mask_to_base64_png(overlay, colorize=False)

        inference_time_ms = (time.time() - start_time) * 1000.0

        logger.info(
            f"Prediction + Risk + Equity + Flood completed in {inference_time_ms:.2f} ms"
        )

        return PredictResponse(
            success=True,
            inference_time_ms=inference_time_ms,
            image_shape=image_rgb.shape,
            mask_shape=pred_mask.shape,
            mask_base64=mask_base64,
            overlay_base64=overlay_base64,
            class_statistics=class_stats,
            unique_classes=sorted(int(c) for c in np.unique(pred_mask)),
            risk_summary=risk_summary,
            risk_tiles=risk_tiles,
            risk_flags=risk_flags,
            timestamp=datetime.now().isoformat(),
            geo_bounds=geo_info.get("bounds"),
            geo_center=geo_info.get("center"),
            has_geo=geo_info.get("has_geo", False),
            equity_summary=equity_summary or None,
            equity_tiles=equity_tiles or None,
            flood_metrics=flood_metrics,
            flood_gif_base64=flood_gif_base64,
            response_plan=response_plan or None,
        )

    except Exception as e:
        logger.error(f"Error in /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
async def evaluate(
    image: UploadFile = File(
        ..., description="Input satellite/aerial image"
    ),
    ground_truth_mask: UploadFile = File(
        ..., description="Ground truth mask (grayscale, class IDs)"
    ),
    metadata: Optional[str] = Form(
        None, description="Optional JSON metadata (pixel_size_meters, etc.)"
    ),
    _: bool = Depends(verify_api_key),
    segmentation_model: SegmentationModel = Depends(ensure_model_loaded),
) -> EvaluateResponse:
    """
    Evaluate model on a single image with ground truth mask.

    Computes:
    - Pixel accuracy and mean pixel accuracy
    - Per-class and mean IoU
    - Dice coefficient
    - Frequency-weighted IoU
    - mAP@50 and mAP@75 (approximated)
    - Confusion matrix
    """
    try:
        start_time = time.time()

        # Parse metadata
        meta_dict: Dict[str, Any] = {}
        pixel_size_meters: Optional[float] = None
        if metadata:
            try:
                meta_dict = json.loads(metadata)
                if meta_dict.get("pixel_size_meters") is not None:
                    pixel_size_meters = float(meta_dict.get("pixel_size_meters"))
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON provided")
            except Exception:
                logger.warning("Invalid pixel_size_meters value in metadata")

        # Load image and mask
        image_bytes = await image.read()
        mask_bytes = await ground_truth_mask.read()

        image_path = save_temporary_file(image_bytes, suffix=".png")
        mask_path = save_temporary_file(mask_bytes, suffix=".png")

        try:
            image_rgb = load_image_as_rgb(image_path)
            gt_mask = load_mask_as_uint8(mask_path)
        finally:
            cleanup_temporary_file(image_path)
            cleanup_temporary_file(mask_path)

        # Run inference
        logger.info(
            f"Running inference for evaluation on image shape {image_rgb.shape}"
        )
        pred_mask = segmentation_model.predict(image_rgb, patch_size=512)

        # ----------------- SANITIZE pred_mask for evaluation path -----------------
        pred_mask = np.asarray(pred_mask)
        if pred_mask.ndim == 4 and pred_mask.shape[0] == 1:
            pred_mask = pred_mask[0]
        if pred_mask.ndim == 3:
            if pred_mask.shape[0] == NUM_CLASSES:
                pred_mask = np.argmax(pred_mask, axis=0).astype(np.uint8)
            elif pred_mask.shape[2] == NUM_CLASSES:
                pred_mask = np.argmax(pred_mask, axis=2).astype(np.uint8)
            else:
                if pred_mask.shape[0] == 1:
                    pred_mask = pred_mask[0]
                elif pred_mask.shape[2] == 1:
                    pred_mask = pred_mask[:, :, 0]
                else:
                    try:
                        pred_mask = np.argmax(pred_mask, axis=0).astype(np.uint8)
                    except Exception:
                        pred_mask = np.squeeze(pred_mask)
        if pred_mask.dtype != np.uint8 and np.issubdtype(pred_mask.dtype, np.floating):
            try:
                if (
                    pred_mask.max() <= 1.0
                    and pred_mask.min() >= 0.0
                    and pred_mask.ndim == 2
                ):
                    pred_mask = (pred_mask > 0.5).astype(np.uint8)
                else:
                    pred_mask = np.round(pred_mask).astype(np.uint8)
            except Exception:
                pred_mask = np.round(pred_mask).astype(np.uint8)
        if pred_mask.ndim != 2:
            pred_mask = np.squeeze(pred_mask)
            if pred_mask.ndim != 2:
                try:
                    pred_mask = np.argmax(pred_mask, axis=-1).astype(np.uint8)
                except Exception:
                    pred_mask = pred_mask.astype(np.uint8)
        # ----------------- END SANITIZE -----------------

        # Ensure masks have same shape
        if pred_mask.shape != gt_mask.shape:
            logger.warning(
                f"Mask shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}. Resizing."
            )
            gt_mask = cv2.resize(
                gt_mask,
                (pred_mask.shape[1], pred_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Compute metrics
        logger.info("Computing evaluation metrics")
        metrics = compute_all_metrics(
            pred_mask,
            gt_mask,
            NUM_CLASSES,
            class_names=list(CLASS_NAMES.values()),
        )

        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(
            gt_mask.flatten(), pred_mask.flatten(), labels=list(range(NUM_CLASSES))
        )
        cm_list = cm.tolist()

        # Compute class statistics
        class_stats = compute_class_statistics(
            pred_mask, pixel_size_meters, CLASS_NAMES
        )

        inference_time_ms = (time.time() - start_time) * 1000.0

        logger.info(f"Evaluation completed in {inference_time_ms:.2f} ms")
        logger.info(f"Mean IoU: {metrics.get('mean_iou', 0.0):.4f}")

        return EvaluateResponse(
            success=True,
            inference_time_ms=inference_time_ms,
            metrics=metrics,
            class_statistics=class_stats,
            confusion_matrix=cm_list,
            unique_pred_classes=sorted(int(c) for c in np.unique(pred_mask)),
            unique_gt_classes=sorted(int(c) for c in np.unique(gt_mask)),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Error in /evaluate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/report", response_model=ReportResponse, tags=["Reporting"])
async def generate_report(
    request: ReportRequest,
    _: bool = Depends(verify_api_key),
) -> ReportResponse:
    """
    Generate an intelligent report using Gemini API.

    Takes evaluation metrics and segmentation summary, calls Gemini API to produce:
    - Executive summary
    - Application-specific insights (Urban Planning, Disaster Management, Automation & Accuracy)
    - Recommendations for model improvement and deployment
    - Markdown report for human-readable output
    """
    try:
        logger.info("Generating report via Gemini API")

        report_result = generate_report_sync(
            metrics_json=request.metrics_json,
            segmentation_summary=request.segmentation_summary,
            context=request.context,
        )

        if report_result.get("status") == "success":
            return ReportResponse(
                success=True,
                status="success",
                report=report_result.get("report"),
                raw_prompt=report_result.get("raw_prompt"),
                raw_response=report_result.get("raw_response"),
                timestamp=datetime.now().isoformat(),
            )
        else:
            return ReportResponse(
                success=False,
                status="error",
                error=report_result.get("message", "Unknown error"),
                timestamp=datetime.now().isoformat(),
            )

    except Exception as e:
        logger.error(f"Error in /report: {e}", exc_info=True)
        return ReportResponse(
            success=False,
            status="error",
            error=str(e),
            timestamp=datetime.now().isoformat(),
        )


# ======================= Main =======================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting FastAPI server on {host}:{port}")
    logger.info(f"Model checkpoint: {MODEL_CHECKPOINT}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Number of classes: {NUM_CLASSES}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
