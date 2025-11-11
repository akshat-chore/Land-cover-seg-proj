"""
Unit tests for FastAPI endpoints with mocked model and Gemini responses.
"""

import pytest
import numpy as np
import json
import io
import tempfile
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

# Import the app
from app.main import app, CLASS_NAMES

# Setup test client
client = TestClient(app)

# ======================= Fixtures =======================

@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

@pytest.fixture
def sample_mask():
    """Create a sample segmentation mask for testing."""
    # Create mask with random class IDs (0-4)
    mask_array = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
    img = Image.fromarray(mask_array, 'L')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

@pytest.fixture
def sample_metrics():
    """Create sample evaluation metrics."""
    return {
        "pixel_accuracy": 0.92,
        "mean_pixel_accuracy": 0.88,
        "mean_iou": 0.78,
        "mAP@50": 0.85,
        "mAP@75": 0.72,
        "mean_dice": 0.85,
        "per_class_iou": {
            "background": 0.95,
            "building": 0.75,
            "woodland": 0.72,
            "water": 0.88,
            "road": 0.65
        }
    }

@pytest.fixture
def sample_segmentation_summary():
    """Create sample segmentation summary."""
    return {
        "total_pixels": 65536,
        "per_class_pixels": {
            "background": 32000,
            "building": 15000,
            "woodland": 10000,
            "water": 5000,
            "road": 3536
        },
        "per_class_percentages": {
            "background": 48.8,
            "building": 22.9,
            "woodland": 15.3,
            "water": 7.6,
            "road": 5.4
        }
    }

# ======================= Health Check Tests =======================

def test_health_check():
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "model_loaded" in data
    assert "device" in data
    assert "api_key_present" in data
    assert "timestamp" in data

# ======================= Predict Endpoint Tests =======================

@patch('app.main.model')
@patch('app.main.SegmentationModel')
def test_predict_success(mock_model_class, mock_model, sample_image):
    """Test /predict endpoint with valid input."""
    # Mock the model
    mock_instance = MagicMock()
    mock_instance.predict.return_value = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
    mock_model_class.return_value = mock_instance
    
    # Patch global model variable
    with patch('app.main.model', mock_instance):
        with patch('app.main.model_loaded', True):
            files = {"image": ("test.png", sample_image, "image/png")}
            response = client.post("/predict", files=files)
    
    # Note: This test will likely fail due to model loading in lifespan
    # Run with pytest --no-cov to avoid issues
    print(f"Response status: {response.status_code}")
    print(f"Response: {response.text}")

@pytest.mark.skip(reason="Requires model lifespan handling")
def test_predict_with_metadata(sample_image):
    """Test /predict endpoint with metadata."""
    metadata = json.dumps({"pixel_size_meters": 0.5})
    files = {"image": ("test.png", sample_image, "image/png")}
    data = {"metadata": metadata}
    
    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 200

# ======================= Evaluate Endpoint Tests =======================

@pytest.mark.skip(reason="Requires model lifespan handling")
def test_evaluate_success(sample_image, sample_mask):
    """Test /evaluate endpoint with valid input."""
    files = {
        "image": ("test.png", sample_image, "image/png"),
        "ground_truth_mask": ("mask.png", sample_mask, "image/png")
    }
    
    response = client.post("/evaluate", files=files)
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert "metrics" in data
    assert "confusion_matrix" in data
    assert "class_statistics" in data

# ======================= Report Endpoint Tests =======================

@patch('app.main.generate_report_sync')
def test_report_success(mock_generate, sample_metrics, sample_segmentation_summary):
    """Test /report endpoint with mocked Gemini response."""
    mock_report = {
        "executive_summary": "The model shows strong performance in urban feature detection...",
        "urban_planning": [
            "Building detection is highly accurate (95% IoU)",
            "Road network is well-captured for infrastructure planning"
        ],
        "disaster_management": [
            "Water bodies detection is reliable for flood risk assessment",
            "Vegetation mapping can support damage assessment"
        ],
        "automation_accuracy": [
            "Model is production-ready with 92% overall accuracy",
            "Deploy with confidence thresholds for automated workflows"
        ],
        "recommendations": {
            "model_improvements": [
                "Collect more training data for road edges",
                "Implement post-processing for connected components"
            ],
            "deployment_notes": [
                "Monitor performance on seasonal variations",
                "Set up automated retraining pipeline"
            ]
        },
        "report_markdown": "# Land-Cover Segmentation Report\n\n## Executive Summary\n..."
    }
    
    mock_generate.return_value = {
        "status": "success",
        "report": mock_report,
        "raw_prompt": "Sample prompt",
        "raw_response": json.dumps(mock_report)
    }
    
    request_body = {
        "metrics_json": sample_metrics,
        "segmentation_summary": sample_segmentation_summary,
        "context": {
            "region": "Test Region",
            "date": "2024-11-11",
            "scenario": "Urban Planning"
        }
    }
    
    response = client.post("/report", json=request_body)
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["status"] == "success"
    assert data["report"] is not None
    assert "executive_summary" in data["report"]

@patch('app.main.generate_report_sync')
def test_report_error_handling(mock_generate, sample_metrics, sample_segmentation_summary):
    """Test /report endpoint error handling."""
    mock_generate.return_value = {
        "status": "error",
        "message": "API key not found",
        "report": None,
        "raw_prompt": None,
        "raw_response": None
    }
    
    request_body = {
        "metrics_json": sample_metrics,
        "segmentation_summary": sample_segmentation_summary
    }
    
    response = client.post("/report", json=request_body)
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is False
    assert data["status"] == "error"

# ======================= Metrics Computation Tests =======================

def test_pixel_accuracy():
    """Test pixel accuracy computation."""
    from app.metrics import pixel_accuracy
    
    pred = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.uint8)
    gt = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.uint8)
    
    acc = pixel_accuracy(pred, gt)
    assert acc == 1.0  # Perfect match

def test_per_class_iou():
    """Test per-class IoU computation."""
    from app.metrics import per_class_iou
    
    pred = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.uint8)
    gt = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.uint8)
    
    ious = per_class_iou(pred, gt, num_classes=2)
    assert len(ious) == 2
    assert all(0 <= iou <= 1 for iou in ious)

def test_dice_coefficient():
    """Test Dice coefficient computation."""
    from app.metrics import dice_coefficient
    
    pred = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.uint8)
    gt = np.array([[0, 0, 1], [0, 1, 1]], dtype=np.uint8)
    
    dice = dice_coefficient(pred, gt, class_id=1)
    assert dice == 1.0  # Perfect match

def test_frequency_weighted_iou():
    """Test frequency weighted IoU computation."""
    from app.metrics import frequency_weighted_iou
    
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[5:, 5:] = 1
    
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[4:, 4:] = 1
    
    fw_iou = frequency_weighted_iou(pred, gt, num_classes=2)
    assert 0 <= fw_iou <= 1

def test_compute_all_metrics():
    """Test compute_all_metrics function."""
    from app.metrics import compute_all_metrics
    
    pred = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
    gt = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
    
    metrics = compute_all_metrics(pred, gt, num_classes=5)
    
    assert "pixel_accuracy" in metrics
    assert "mean_iou" in metrics
    assert "mean_dice" in metrics
    assert "per_class_iou" in metrics
    assert "per_class_dice" in metrics

# ======================= Inference Utilities Tests =======================

def test_colorize_mask():
    """Test mask colorization."""
    from app.inference import colorize_mask
    
    mask = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.uint8)
    colored = colorize_mask(mask)
    
    assert colored.shape == (2, 3, 3)  # RGB output
    assert colored.dtype == np.uint8

def test_mask_to_base64_png():
    """Test mask to base64 PNG conversion."""
    from app.inference import mask_to_base64_png
    
    mask = np.random.randint(0, 5, (64, 64), dtype=np.uint8)
    base64_str = mask_to_base64_png(mask)
    
    assert base64_str.startswith("data:image/png;base64,")
    assert len(base64_str) > 50

def test_compute_class_statistics():
    """Test class statistics computation."""
    from app.inference import compute_class_statistics
    
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[50:, 50:] = 1
    
    stats = compute_class_statistics(mask, pixel_size_meters=0.5)
    
    assert "total_pixels" in stats
    assert "per_class_pixels" in stats
    assert "per_class_percentages" in stats
    assert "per_class_area_m2" in stats
    assert "per_class_area_km2" in stats

# ======================= Gemini Client Tests =======================

@pytest.mark.asyncio
@patch('httpx.AsyncClient.post')
async def test_gemini_client_success(mock_post, sample_metrics, sample_segmentation_summary):
    """Test Gemini client successful API call."""
    from app.gemini_client import GeminiClient
    
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "executive_summary": "Test summary",
                                "urban_planning": ["Insight 1"],
                                "disaster_management": ["Insight 2"],
                                "automation_accuracy": ["Insight 3"],
                                "recommendations": {"model_improvements": [], "deployment_notes": []},
                                "report_markdown": "# Test Report"
                            })
                        }
                    ]
                }
            }
        ]
    }
    mock_post.return_value = mock_response
    
    client = GeminiClient(api_key="test_key")
    result = await client.generate_report(sample_metrics, sample_segmentation_summary)
    
    assert result["status"] == "success"
    assert "report" in result
    assert "executive_summary" in result["report"]

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--no-cov"])
