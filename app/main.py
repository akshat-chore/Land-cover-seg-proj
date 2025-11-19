"""
FastAPI server for land-cover semantic segmentation.
Provides endpoints for prediction, evaluation, and report generation.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import time
import json
from contextlib import asynccontextmanager
from dotenv import load_dotenv

import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
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
    load_image_as_rgb, load_mask_as_uint8, colorize_mask, overlay_mask_on_image,
    mask_to_base64_png, compute_class_statistics, save_temporary_file, cleanup_temporary_file,
    compute_risk_map  # NEW: compute risk map and tiles
)

# ======================= Logging Setup =======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# ======================= Configuration =======================
MODEL_CHECKPOINT = os.getenv('MODEL_CHECKPOINT', 'models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth')
DEVICE = os.getenv('DEVICE', 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu')
NUM_CLASSES = int(os.getenv('NUM_CLASSES', '5'))
ENCODER = os.getenv('ENCODER', 'efficientnet-b0')
ENCODER_WEIGHTS = os.getenv('ENCODER_WEIGHTS', 'imagenet')
SERVER_API_KEY = os.getenv('SERVER_API_KEY', None)  # Optional API key for server
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', None)  # Gemini API key for report generation
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

# Class names mapping
CLASS_NAMES = {
    0: 'background',
    1: 'building',
    2: 'woodland',
    3: 'water',
    4: 'road'
}

# ======================= Global State =======================
model: Optional[SegmentationModel] = None
model_loaded: bool = False
device_info: str = DEVICE

# ======================= Model Initialization =======================
def load_model_on_startup():
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
        
        print(f"[STARTUP] Initializing SegmentationModel...")
        model = SegmentationModel(
            checkpoint_path=MODEL_CHECKPOINT,
            encoder=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            num_classes=NUM_CLASSES,
            device=DEVICE
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
    description="API for land-cover semantic segmentation with evaluation metrics and Gemini-powered reports",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ['*'] else ['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================= Dependencies =======================
def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """Optional API key verification."""
    if SERVER_API_KEY:
        if x_api_key != SERVER_API_KEY:
            raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key header")
    return True

def ensure_model_loaded():
    """Ensure model is loaded before processing.""" 
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server startup logs.")
    return model

# ======================= Pydantic Models =======================
class PredictResponse(BaseModel):
    """Response model for /predict endpoint."""
    success: bool
    inference_time_ms: float
    image_shape: tuple
    mask_shape: tuple
    mask_base64: str = Field(..., description="Base64-encoded colorized mask PNG")
    overlay_base64: str = Field(..., description="Base64-encoded overlay PNG")
    class_statistics: Dict[str, Any]
    unique_classes: List[int] = Field(..., description="Classes found in prediction")
    # NEW fields for risk
    risk_summary: Optional[Dict[str, Any]] = Field(None, description="Summary metrics for risk")
    risk_tiles: Optional[List[Dict[str, Any]]] = Field(None, description="Tile-level risk scores")
    risk_flags: Optional[List[str]] = Field(None, description="Flags like roads_missing, high_flood_exposure")
    timestamp: str

        # ADD THESE NEW FIELDS:
    geo_bounds: Optional[List[float]] = Field(None, description="[lat_min, lon_min, lat_max, lon_max]")
    geo_center: Optional[List[float]] = Field(None, description="[lat, lon] center point")
    has_geo: bool = Field(False, description="Whether image has geographic coordinates")

class EvaluateResponse(BaseModel):
    """Response model for /evaluate endpoint."""
    success: bool
    inference_time_ms: float
    metrics: Dict[str, Any]
    class_statistics: Dict[str, Any]
    confusion_matrix: list  # 2D array
    unique_pred_classes: list
    unique_gt_classes: list
    timestamp: str

class ReportRequest(BaseModel):
    """Request model for /report endpoint.""" 
    metrics_json: Dict[str, Any] = Field(..., description="Metrics from evaluation")
    segmentation_summary: Dict[str, Any] = Field(..., description="Summary of segmentation results")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional context (region, date, scenario)")

class ReportResponse(BaseModel):
    """Response model for /report endpoint."""
    success: bool
    status: str
    report: Optional[Dict[str, Any]] = Field(None, description="Generated report with insights")
    raw_prompt: Optional[str] = Field(None, description="Exact prompt sent to Gemini")
    raw_response: Optional[str] = Field(None, description="Raw Gemini response")
    error: Optional[str] = Field(None, description="Error message if generation failed")
    timestamp: str

class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    model_loaded: bool
    device: str
    api_key_present: bool
    timestamp: str

# ======================= Endpoints =======================

@app.get("/", tags=["Server"])
async def root():
    """
    Root endpoint - redirects to API documentation.
    """
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
        timestamp=datetime.now().isoformat()
    )

def extract_geotiff_bounds(image_path: str):
    """
    Extract geographic bounds from a GeoTIFF file.
    Returns lat/lon bounds and a function to transform pixel coords to lat/lon.
    """
    try:
        with rasterio.open(image_path) as src:
            # Get bounds in original CRS (EPSG:2180 for your dataset)
            bounds = src.bounds  # (left, bottom, right, top)
            original_crs = src.crs
            
            # Transform to WGS84 (lat/lon) for web mapping
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                original_crs, 'EPSG:4326', 
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )
            
            # Get image dimensions
            height, width = src.height, src.width
            
            # Create transform function for pixel -> lat/lon
            def pixel_to_latlon(x, y):
                """Convert pixel coordinates to lat/lon"""
                # Calculate fractional position
                lon = lon_min + (x / width) * (lon_max - lon_min)
                lat = lat_max - (y / height) * (lat_max - lat_min)  # Note: y is inverted
                return lat, lon
            
            return {
                'bounds': [lat_min, lon_min, lat_max, lon_max],  # [south, west, north, east]
                'center': [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
                'pixel_to_latlon': pixel_to_latlon,
                'has_geo': True
            }
    except Exception as e:
        logger.warning(f"Could not extract geo info: {e}")
        return {'has_geo': False}

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(
    image: UploadFile = File(..., description="Input satellite/aerial image (PNG, JPG, TIFF)"),
    metadata: Optional[str] = Form(None, description="Optional JSON metadata (pixel_size_meters, etc.)"),
    confidence_threshold: float = Form(0.5, description="Confidence threshold (not used in semantic segmentation but kept for compatibility)"),
    _: bool = Depends(verify_api_key),
    segmentation_model: SegmentationModel = Depends(ensure_model_loaded)
) -> PredictResponse:
    """
    Run inference on a single satellite image.
    
    Returns:
    - Segmentation mask (base64 PNG)
    - Colorized overlay (base64 PNG)
    - Per-class statistics (pixel counts, percentages, areas if pixel_size provided)
    - Inference time
    - Urban risk analysis (tile-level risk, summary, flags)
    """
    try:
        start_time = time.time()
        
        # Parse metadata if provided
        meta_dict = {}
        pixel_size_meters = None
        if metadata:
            try:
                meta_dict = json.loads(metadata)
                pixel_size_meters = float(meta_dict.get('pixel_size_meters')) if meta_dict.get('pixel_size_meters') else None
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON provided")
            except Exception:
                logger.warning("Invalid pixel_size_meters value in metadata")
        
        # Save and load image
        image_bytes = await image.read()
        image_path = save_temporary_file(image_bytes, suffix='.png')
        geo_info = {'has_geo': False}  # Default
        try:
            geo_info = extract_geotiff_bounds(image_path)
            image_rgb = load_image_as_rgb(image_path)
        finally:
            pass
            
        
        # Run inference
        logger.info(f"Running inference on image shape {image_rgb.shape}")
        pred_mask = segmentation_model.predict(image_rgb, patch_size=512)

        # ----------------- SANITIZE pred_mask to a 2D integer label mask -----------------
        # ensure numpy array
        pred_mask = np.asarray(pred_mask)

        # If model returned a batch dim (1, ...)
        if pred_mask.ndim == 4 and pred_mask.shape[0] == 1:
            pred_mask = pred_mask[0]

        # If model returned (C, H, W) probability map -> argmax over channel
        if pred_mask.ndim == 3:
            # common cases: (C, H, W) or (H, W, C)
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
            # If probabilities in 0..1 and only one channel, threshold at 0.5
            try:
                if pred_mask.max() <= 1.0 and pred_mask.min() >= 0.0 and pred_mask.ndim == 2:
                    pred_mask = (pred_mask > 0.5).astype(np.uint8)
                else:
                    pred_mask = np.round(pred_mask).astype(np.uint8)
            except Exception:
                pred_mask = np.round(pred_mask).astype(np.uint8)

        # Ensure it's 2D
        if pred_mask.ndim != 2:
            pred_mask = np.squeeze(pred_mask)
            if pred_mask.ndim != 2:
                # last resort: take argmax over any channel axis
                try:
                    pred_mask = np.argmax(pred_mask, axis=-1).astype(np.uint8)
                except Exception:
                    pred_mask = pred_mask.astype(np.uint8)
        # ----------------- END SANITIZE -----------------

        # Colorize mask and create overlay
        colored_mask = colorize_mask(pred_mask)
        overlay = overlay_mask_on_image(image_rgb, pred_mask, alpha=0.5)
        
        # Compute class statistics
        class_stats = compute_class_statistics(pred_mask, pixel_size_meters, CLASS_NAMES)
        
        # 🧮 Compute Urban Risk Map and Tile Scores (new)
        try:
            risk_map, risk_tiles, risk_summary, risk_flags = compute_risk_map(
                pred_mask,
                class_names_map=CLASS_NAMES,
                tile_size=256,
                pixel_size_meters=pixel_size_meters
            )
            if geo_info.get('has_geo') and 'pixel_to_latlon' in geo_info:
                 for tile in risk_tiles:
                     
                    center_x = tile['x'] + tile['w'] / 2
                    center_y = tile['y'] + tile['h'] / 2
                    lat, lon = geo_info['pixel_to_latlon'](center_x, center_y)
                    tile['lat'] = lat
                    tile['lon'] = lon
            
        except Exception as re:
            logger.exception("Error computing risk map, skipping risk outputs: %s", re)
            risk_map, risk_tiles, risk_summary, risk_flags = None, [], {}, []
        finally:
            # Cleanup temp file now
            cleanup_temporary_file(image_path)

        # Convert to base64
        mask_base64 = mask_to_base64_png(pred_mask, colorize=True)
        overlay_base64 = mask_to_base64_png(overlay, colorize=False)
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Prediction + Risk completed in {inference_time_ms:.2f} ms")
        
        return PredictResponse(
            success=True,
            inference_time_ms=inference_time_ms,
            image_shape=image_rgb.shape,
            mask_shape=pred_mask.shape,
            mask_base64=mask_base64,
            overlay_base64=overlay_base64,
            class_statistics=class_stats,
            unique_classes=sorted([int(c) for c in np.unique(pred_mask)]),
            risk_summary=risk_summary,
            risk_tiles=risk_tiles,
            risk_flags=risk_flags,
            timestamp=datetime.now().isoformat(),
            geo_bounds=geo_info.get('bounds'),  # ADD THIS
            geo_center=geo_info.get('center'),  # ADD THIS
            has_geo=geo_info.get('has_geo', False)  # ADD THIS
        )
    
    except Exception as e:
        logger.error(f"Error in /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
async def evaluate(
    image: UploadFile = File(..., description="Input satellite/aerial image"),
    ground_truth_mask: UploadFile = File(..., description="Ground truth mask (grayscale, class IDs)"),
    metadata: Optional[str] = Form(None, description="Optional JSON metadata"),
    _: bool = Depends(verify_api_key),
    segmentation_model: SegmentationModel = Depends(ensure_model_loaded)
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
        meta_dict = {}
        pixel_size_meters = None
        if metadata:
            try:
                meta_dict = json.loads(metadata)
                pixel_size_meters = meta_dict.get('pixel_size_meters')
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON provided")
        
        # Load image and mask
        image_bytes = await image.read()
        mask_bytes = await ground_truth_mask.read()
        
        image_path = save_temporary_file(image_bytes, suffix='.png')
        mask_path = save_temporary_file(mask_bytes, suffix='.png')
        
        try:
            image_rgb = load_image_as_rgb(image_path)
            gt_mask = load_mask_as_uint8(mask_path)
        finally:
            cleanup_temporary_file(image_path)
            cleanup_temporary_file(mask_path)
        
        # Run inference
        logger.info(f"Running inference for evaluation on image shape {image_rgb.shape}")
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
                if pred_mask.max() <= 1.0 and pred_mask.min() >= 0.0 and pred_mask.ndim == 2:
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
            logger.warning(f"Mask shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}. Resizing...")
            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Compute metrics
        logger.info("Computing evaluation metrics")
        metrics = compute_all_metrics(pred_mask, gt_mask, NUM_CLASSES, 
                                      class_names=list(CLASS_NAMES.values()))
        
        # Compute confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(gt_mask.flatten(), pred_mask.flatten(), 
                             labels=list(range(NUM_CLASSES)))
        cm_list = cm.tolist()
        
        # Compute class statistics
        class_stats = compute_class_statistics(pred_mask, pixel_size_meters, CLASS_NAMES)
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Evaluation completed in {inference_time_ms:.2f} ms")
        logger.info(f"Mean IoU: {metrics['mean_iou']:.4f}")
        
        return EvaluateResponse(
            success=True,
            inference_time_ms=inference_time_ms,
            metrics=metrics,
            class_statistics=class_stats,
            confusion_matrix=cm_list,
            unique_pred_classes=sorted([int(c) for c in np.unique(pred_mask)]),
            unique_gt_classes=sorted([int(c) for c in np.unique(gt_mask)]),
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error in /evaluate: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report", response_model=ReportResponse, tags=["Reporting"])
async def generate_report(
    request: ReportRequest,
    _: bool = Depends(verify_api_key)
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
            context=request.context
        )
        
        if report_result['status'] == 'success':
            return ReportResponse(
                success=True,
                status='success',
                report=report_result.get('report'),
                raw_prompt=report_result.get('raw_prompt'),
                raw_response=report_result.get('raw_response'),
                timestamp=datetime.now().isoformat()
            )
        else:
            return ReportResponse(
                success=False,
                status='error',
                error=report_result.get('message', 'Unknown error'),
                timestamp=datetime.now().isoformat()
            )
    
    except Exception as e:
        logger.error(f"Error in /report: {e}", exc_info=True)
        return ReportResponse(
            success=False,
            status='error',
            error=str(e),
            timestamp=datetime.now().isoformat()
        )

# ======================= Main =======================
if __name__ == "__main__":
    port = int(os.getenv('PORT', '8000'))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info(f"Starting FastAPI server on {host}:{port}")
    logger.info(f"Model checkpoint: {MODEL_CHECKPOINT}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Number of classes: {NUM_CLASSES}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
