# Quick Start Guide

## 5-Minute Setup

### Option 1: Local Python Environment

```bash
# 1. Navigate to project directory
cd Land-Cover-Semantic-Segmentation-PyTorch

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set API key
export GEMINI_API_KEY=your_key_here

# 5. Start server
python -m uvicorn app.main:app --reload
```

### Option 2: Docker

```bash
# 1. Build image
docker build -t land-cover-api .

# 2. Run container
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=your_key_here \
  land-cover-api
```

### Option 3: Docker Compose

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env with your settings
# Set GEMINI_API_KEY=your_key_here

# 3. Start with compose
docker-compose up -d
```

---

## Verify Server is Running

```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "api_key_present": true
}
```

---

## Test Endpoints

### 1. Predict on an Image

```bash
# Using a sample image
curl -X POST http://localhost:8000/predict \
  -F "image=@path/to/satellite_image.png"
```

### 2. Evaluate with Ground Truth

```bash
curl -X POST http://localhost:8000/evaluate \
  -F "image=@satellite_image.png" \
  -F "ground_truth_mask=@ground_truth_mask.png"
```

### 3. Generate AI Report

```bash
cat > report_request.json << 'EOF'
{
  "metrics_json": {
    "pixel_accuracy": 0.92,
    "mean_iou": 0.78,
    "mean_dice": 0.85,
    "per_class_iou": {
      "background": 0.95,
      "building": 0.75,
      "woodland": 0.72,
      "water": 0.88,
      "road": 0.65
    }
  },
  "segmentation_summary": {
    "total_pixels": 262144,
    "per_class_pixels": {
      "background": 128000,
      "building": 50000,
      "woodland": 40000,
      "water": 30000,
      "road": 14144
    },
    "per_class_percentages": {
      "background": 48.8,
      "building": 19.1,
      "woodland": 15.3,
      "water": 11.5,
      "road": 5.4
    }
  },
  "context": {
    "region": "Downtown Manhattan",
    "date": "2024-11-11",
    "scenario": "Urban planning assessment"
  }
}
EOF

curl -X POST http://localhost:8000/report \
  -H "Content-Type: application/json" \
  -d @report_request.json
```

---

## Access Interactive Documentation

Open browser to: `http://localhost:8000/docs`

This provides an interactive Swagger UI where you can:
- Test all endpoints
- See request/response schemas
- Try different parameter combinations

---

## Common Commands

### View Logs (Docker)

```bash
docker-compose logs -f land-cover-api
```

### Stop Server (Docker)

```bash
docker-compose down
```

### Remove GPU Usage Restriction

```bash
# If running into memory issues, use CPU
docker-compose down
docker-compose -f docker-compose.yml run -e DEVICE=cpu land-cover-api
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Model not loaded` | Check `MODEL_CHECKPOINT` path is correct |
| `API key error` | Verify `GEMINI_API_KEY` is set and valid |
| `Out of memory` | Use CPU (`DEVICE=cpu`) or reduce batch size |
| `Port already in use` | Change port: `docker-compose.yml` or `-p 8001:8000` |

---

## Next Steps

1. **Review the full API documentation**: `API_README.md`
2. **Explore test endpoints**: `tests/test_endpoints.py`
3. **Customize class mapping**: `app/inference.py` (CLASS_COLORS, CLASS_NAMES)
4. **Integrate with your frontend**: See endpoint specifications in `API_README.md`
5. **Deploy to production**: Use docker-compose with proper scaling

---

## Getting Help

- **API Docs**: `http://localhost:8000/docs`
- **Issues**: Check troubleshooting section in `API_README.md`
- **Code Examples**: See test file `tests/test_endpoints.py`
