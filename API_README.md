# Land-Cover Semantic Segmentation FastAPI Server

A production-ready FastAPI server for land-cover semantic segmentation with integrated Gemini AI report generation. This server runs inference on satellite/aerial images, computes comprehensive evaluation metrics, and generates intelligent reports for urban planning, disaster management, and automation use cases.

## Features

- **🚀 High-Performance Inference**: Tile-based inference for large satellite images with seamless reconstruction
- **📊 Comprehensive Metrics**: Pixel accuracy, per-class IoU, Dice coefficient, mAP@50/mAP@75, and more
- **🤖 AI-Powered Reports**: Automatic report generation using Google Gemini API with application-specific insights
- **🏙️ Multi-Domain Applications**:
  - Urban Planning (building detection, infrastructure mapping)
  - Disaster Management (flood/fire risk assessment, damage mapping)
  - Automation & Accuracy (model performance analysis, deployment readiness)
- **📈 RESTful API**: FastAPI with automatic interactive documentation (Swagger UI)
- **🔒 Security**: Optional API key authentication, CORS support
- **🐳 Docker**: Ready-to-deploy containerization
- **✅ Well-Tested**: Unit tests with mocked model and Gemini responses

## Architecture

```
app/
├── main.py              # FastAPI application with all endpoints
├── model.py             # PyTorch model loading and inference
├── metrics.py           # Evaluation metrics computation
├── gemini_client.py     # Gemini API integration for report generation
├── inference.py         # Image preprocessing and output utilities
├── utils.py             # Common utilities (constants, logging)
└── __init__.py          # Package initialization

tests/
├── test_endpoints.py    # Unit tests with mocked components
└── __init__.py

Dockerfile              # Docker image configuration
requirements.txt        # Python dependencies
README.md              # This file
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, falls back to CPU)
- PyTorch 2.0+
- Google Gemini API key (for report generation)

### Local Setup

1. **Clone the repository** (or extract the project)
   ```bash
   cd Land-Cover-Semantic-Segmentation-PyTorch
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   export MODEL_CHECKPOINT=models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth
   export GEMINI_API_KEY=your_api_key_here
   export DEVICE=cuda  # or 'cpu'
   export NUM_CLASSES=5
   export ENCODER=efficientnet-b0
   export ENCODER_WEIGHTS=imagenet
   export PORT=8000
   export HOST=0.0.0.0
   ```

5. **Start the server**
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

   The API will be available at `http://localhost:8000`

### Docker Setup

1. **Build the Docker image**
   ```bash
   docker build -t land-cover-segmentation:latest .
   ```

2. **Run the container**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e MODEL_CHECKPOINT=/app/models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth \
     -e GEMINI_API_KEY=your_api_key_here \
     -e DEVICE=cuda \
     -v $(pwd)/models:/app/models \
     --gpus all \
     --name land-cover-seg \
     land-cover-segmentation:latest
   ```

   Or without GPU:
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e DEVICE=cpu \
     -v $(pwd)/models:/app/models \
     --name land-cover-seg \
     land-cover-segmentation:latest
   ```

## API Endpoints

### 1. **GET /health**
Health check endpoint returning server status and model availability.

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "api_key_present": true,
  "timestamp": "2024-11-11T10:30:45.123456"
}
```

---

### 2. **POST /predict**
Run inference on a satellite image and return the segmentation mask.

**Parameters:**
- `image` (file, required): Satellite/aerial image (PNG, JPG, TIFF)
- `metadata` (JSON, optional): Metadata including `pixel_size_meters` for area computation
- `confidence_threshold` (float, optional): Confidence threshold (default: 0.5)
- `X-API-Key` (header, optional): API key if `SERVER_API_KEY` is set

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@satellite_image.png" \
  -F 'metadata={"pixel_size_meters": 0.5}'
```

**Response:**
```json
{
  "success": true,
  "inference_time_ms": 234.5,
  "image_shape": [512, 512, 3],
  "mask_shape": [512, 512],
  "mask_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "overlay_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "class_statistics": {
    "total_pixels": 262144,
    "per_class_pixels": {
      "background": 130000,
      "building": 50000,
      "woodland": 40000,
      "water": 25000,
      "road": 17144
    },
    "per_class_percentages": {
      "background": 49.6,
      "building": 19.1,
      "woodland": 15.3,
      "water": 9.5,
      "road": 6.5
    },
    "per_class_area_m2": {
      "background": 32500000.0,
      "building": 12500000.0,
      "woodland": 10000000.0,
      "water": 6250000.0,
      "road": 4286000.0
    },
    "per_class_area_km2": {
      "background": 32.5,
      "building": 12.5,
      "woodland": 10.0,
      "water": 6.25,
      "road": 4.286
    }
  },
  "unique_classes": [0, 1, 2, 3, 4],
  "timestamp": "2024-11-11T10:35:20.654321"
}
```

---

### 3. **POST /evaluate**
Run inference and compare with ground truth mask to compute comprehensive metrics.

**Parameters:**
- `image` (file, required): Satellite/aerial image
- `ground_truth_mask` (file, required): Ground truth mask (grayscale, class IDs 0-255)
- `metadata` (JSON, optional): Optional metadata
- `X-API-Key` (header, optional): API key if required

**Example:**
```bash
curl -X POST http://localhost:8000/evaluate \
  -F "image=@satellite_image.png" \
  -F "ground_truth_mask=@ground_truth_mask.png"
```

**Response:**
```json
{
  "success": true,
  "inference_time_ms": 450.2,
  "metrics": {
    "pixel_accuracy": 0.92,
    "mean_pixel_accuracy": 0.88,
    "mean_iou": 0.78,
    "frequency_weighted_iou": 0.82,
    "mean_dice": 0.85,
    "mAP@50": 0.85,
    "mAP@75": 0.72,
    "per_class_iou": {
      "background": 0.95,
      "building": 0.75,
      "woodland": 0.72,
      "water": 0.88,
      "road": 0.65
    },
    "per_class_dice": {
      "background": 0.97,
      "building": 0.86,
      "woodland": 0.84,
      "water": 0.94,
      "road": 0.79
    },
    "per_class_ap_50": {
      "background": 0.98,
      "building": 0.80,
      "woodland": 0.75,
      "water": 0.90,
      "road": 0.72
    },
    "per_class_ap_75": {
      "background": 0.95,
      "building": 0.65,
      "woodland": 0.60,
      "water": 0.80,
      "road": 0.55
    }
  },
  "class_statistics": {...},
  "confusion_matrix": [[950, 30, 15, 5, 0], ...],
  "unique_pred_classes": [0, 1, 2, 3, 4],
  "unique_gt_classes": [0, 1, 2, 3, 4],
  "timestamp": "2024-11-11T10:36:15.789012"
}
```

---

### 4. **POST /report**
Generate an intelligent report using Gemini API with application-specific insights.

**Request Body:**
```json
{
  "metrics_json": {
    "pixel_accuracy": 0.92,
    "mean_iou": 0.78,
    "per_class_iou": {...}
  },
  "segmentation_summary": {
    "total_pixels": 262144,
    "per_class_pixels": {...},
    "per_class_percentages": {...}
  },
  "context": {
    "region": "Downtown Manhattan",
    "date": "2024-11-11",
    "scenario": "Urban development assessment",
    "image_resolution_m": 0.5
  }
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/report \
  -H "Content-Type: application/json" \
  -d @report_request.json
```

**Response:**
```json
{
  "success": true,
  "status": "success",
  "report": {
    "executive_summary": "The model demonstrates strong performance in urban feature detection with 92% overall accuracy. Building identification is particularly accurate (95% IoU), making it suitable for infrastructure planning. Water bodies are reliably detected (88% IoU) for flood risk assessment.",
    "urban_planning": [
      "Building detection accuracy of 95% enables reliable infrastructure inventory for urban planning",
      "Road network segmentation (79% Dice) can support transportation planning and maintenance scheduling",
      "High precision (98% AP@50) in background classification ensures minimal false positives in building identification"
    ],
    "disaster_management": [
      "Water detection (94% Dice) provides reliable baseline for flood risk mapping and early warning systems",
      "Vegetation detection enables rapid damage assessment for forests and parks affected by disasters",
      "Multi-class segmentation allows rapid post-disaster analysis of affected land-use categories"
    ],
    "automation_accuracy": [
      "Model shows production-ready performance with 92% pixel accuracy; recommend deployment with automated thresholds",
      "Per-class metrics are consistent, suggesting reliable generalization across different regions",
      "Frequency-weighted IoU (82%) indicates good balance across common land-cover types"
    ],
    "recommendations": {
      "model_improvements": [
        "Collect additional training data for underperforming classes (road: 65% IoU)",
        "Implement morphological post-processing to improve road connectivity",
        "Consider data augmentation with seasonal variations for robust performance"
      ],
      "deployment_notes": [
        "Monitor performance on new imagery; set up automated retraining pipeline",
        "Deploy with confidence thresholds per class for production accuracy targets",
        "Implement tile caching for repeated analysis of same regions"
      ]
    },
    "report_markdown": "# Land-Cover Segmentation Analysis Report\n\n## Executive Summary\n...[Full markdown report with tables and visualizations]..."
  },
  "raw_prompt": "You are an expert in remote sensing...[Full system prompt sent to Gemini]...",
  "raw_response": "{...Raw JSON response from Gemini API...}",
  "timestamp": "2024-11-11T10:37:30.234567"
}
```

---

### 5. **GET /docs**
Interactive API documentation (Swagger UI).

**Access:** `http://localhost:8000/docs`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CHECKPOINT` | `models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth` | Path to model checkpoint |
| `GEMINI_API_KEY` | (required for `/report`) | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model identifier |
| `DEVICE` | `cuda` or `cpu` | PyTorch device (auto-detects GPU) |
| `NUM_CLASSES` | `5` | Number of segmentation classes |
| `ENCODER` | `efficientnet-b0` | Encoder backbone name |
| `ENCODER_WEIGHTS` | `imagenet` | Pre-trained weights for encoder |
| `SERVER_API_KEY` | (optional) | API key for server authentication |
| `CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=app --cov-report=html

# Run specific test
pytest tests/test_endpoints.py::test_health_check -v
```

## Class Mapping

Default class mapping (customizable in `app/inference.py`):

| Class ID | Class Name | Color (RGB) |
|----------|-----------|------------|
| 0 | Background | Black (0, 0, 0) |
| 1 | Building | Red (200, 0, 0) |
| 2 | Woodland | Green (34, 139, 34) |
| 3 | Water | Blue (0, 149, 218) |
| 4 | Road | Gray (128, 128, 128) |

Modify `CLASS_COLORS` and `CLASS_NAMES` in `app/inference.py` to customize.

## Production Deployment

### Using Gunicorn + Uvicorn

```bash
pip install gunicorn

gunicorn app.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Docker Compose Example

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    container_name: land-cover-seg
    ports:
      - "8000:8000"
    environment:
      - MODEL_CHECKPOINT=/app/models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - DEVICE=cuda
      - NUM_CLASSES=5
    volumes:
      - ./models:/app/models
      - ./workdir:/app/workdir
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

Run with:
```bash
GEMINI_API_KEY=your_key docker-compose up -d
```

## Common Issues & Troubleshooting

### 1. Model not loading
**Issue:** `Model checkpoint file not found`
**Solution:** Ensure `MODEL_CHECKPOINT` env var points to the correct model path.

### 2. CUDA out of memory
**Issue:** `RuntimeError: CUDA out of memory`
**Solution:** 
- Reduce patch size: Modify `predict()` call in endpoints with smaller `patch_size`
- Use CPU: Set `DEVICE=cpu`

### 3. Gemini API errors
**Issue:** `ValueError: Invalid GEMINI_API_KEY`
**Solution:**
- Verify API key is correct and active
- Check rate limits on Google Cloud Console
- Ensure model exists (default: `gemini-1.5-flash`)

### 4. Permission denied on model checkpoint
**Issue:** `PermissionError: [Errno 13] Permission denied`
**Solution:** Ensure model file is readable:
```bash
chmod 644 models/trained_landcover_unet_efficientnet-b0_epochs18_patch512_batch16.pth
```

## TODO & Future Improvements

- [ ] Add batch processing endpoint for multiple images
- [ ] Implement caching with Redis for repeated predictions
- [ ] Add visualization dashboard with Streamlit/Plotly
- [ ] Support for multi-model ensemble predictions
- [ ] Real-time inference streaming with WebSockets
- [ ] Automated model retraining pipeline integration
- [ ] Explainability features (GradCAM, attention maps)
- [ ] Support for additional encoders (ResNet, ViT, etc.)
- [ ] Performance profiling and optimization

## Citation

If you use this project in your research, please cite:

```bibtex
@software{land_cover_seg_2024,
  title={Land-Cover Semantic Segmentation FastAPI Server},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/land-cover-segmentation}
}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Support & Contact

- **Issues**: Open an issue on GitHub
- **Documentation**: Available at `http://localhost:8000/docs` when server is running
- **API Schema**: Available at `http://localhost:8000/openapi.json`

---

**Happy segmenting! 🚀**
