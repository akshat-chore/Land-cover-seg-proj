#!/usr/bin/env python3
"""
Debug the exact JSON parsing issue.
"""

import json
import httpx

GEMINI_API_KEY = "AIzaSyArjwEY4xcmJA6YrCl2r7DkGTPJ5zFWHss"
MODEL = "gemini-2.5-flash"

metrics = {
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

summary = {
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
    }
}

context = {
    "region": "Downtown Manhattan",
    "date": "2024-09-15",
    "scenario": "Urban development assessment"
}

prompt = f"""You are an expert in remote sensing, land cover classification, and geospatial analysis.

A semantic segmentation model has been applied to satellite/aerial imagery. Below are the results:

**Evaluation Metrics (Model Performance):**
{json.dumps(metrics, indent=2)}

**Segmentation Summary (Areas & Percentages):**
{json.dumps(summary, indent=2)}

**Context Information:**
{json.dumps(context, indent=2) if context else "No context provided."}

Please generate a detailed report in JSON format with the following structure:
{{
    "executive_summary": "A 3-4 sentence summary of key findings.",
    "urban_planning": [
        "Insight 1 relevant to urban planning with recommended action",
        "Insight 2...",
        "..."
    ],
    "disaster_management": [
        "Insight 1 relevant to disaster risk management",
        "Insight 2...",
        "..."
    ],
    "automation_accuracy": [
        "Assessment of model automation readiness",
        "Recommendations for production deployment",
        "Data quality observations"
    ],
    "recommendations": {{
        "model_improvements": [
            "Data augmentation suggestions",
            "Architecture/training recommendations",
            "..."
        ],
        "deployment_notes": [
            "Production readiness notes",
            "Monitoring recommendations",
            "..."
        ]
    }}
}}

Ensure the JSON is valid and can be parsed. Return ONLY the JSON object in markdown code blocks, nothing else."""

print("Sending full prompt to Gemini API...")

request_body = {
    "contents": [{
        "parts": [{"text": prompt}]
    }],
    "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 4096
    }
}

try:
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}",
        json=request_body,
        timeout=60.0
    )
    
    if response.status_code == 200:
        result = response.json()
        generated_text = result['candidates'][0]['content']['parts'][0]['text']
        
        # Apply the extraction logic
        extracted_text = generated_text.strip()
        
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text[7:]
            if extracted_text.endswith("```"):
                extracted_text = extracted_text[:-3]
        elif extracted_text.startswith("```"):
            extracted_text = extracted_text[3:]
            if extracted_text.endswith("```"):
                extracted_text = extracted_text[:-3]
        
        extracted_text = extracted_text.strip()
        
        print("Attempting to parse JSON...")
        try:
            report_json = json.loads(extracted_text)
            print("✓ SUCCESS!")
            print(json.dumps({k: type(v).__name__ for k, v in report_json.items()}, indent=2))
        except json.JSONDecodeError as e:
            print(f"✗ JSON Parse Error: {e}")
            print(f"Error at position {e.pos}")
            print(f"Line {e.lineno}, Column {e.colno}")
            
            # Show context around error
            lines = extracted_text.split('\n')
            if e.lineno <= len(lines):
                error_line = lines[e.lineno - 1]
                print(f"\nError line ({e.lineno}):")
                print(error_line)
                print(" " * (e.colno - 1) + "^")
                
            # Try with jsoncomment (if available)
            try:
                print("\nAttempting with json5 parsing...")
                import json5
                report_json = json5.loads(extracted_text)
                print("✓ json5 parse succeeded!")
            except ImportError:
                print("(json5 not available)")
            except Exception as e2:
                print(f"✗ json5 also failed: {e2}")
    else:
        print(f"API Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
