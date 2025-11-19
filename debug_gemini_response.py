#!/usr/bin/env python3
"""
Debug script to check what Gemini is actually returning for the report prompt.
"""

import os
import json
import httpx

# Use the API key from your logs
GEMINI_API_KEY = "AIzaSyArjwEY4xcmJA6YrCl2r7DkGTPJ5zFWHss"
MODEL = "gemini-2.5-flash"

metrics = {
    "pixel_accuracy": 0.92,
    "mean_pixel_accuracy": 0.88,
    "mean_iou": 0.78,
    "per_class_iou": {"background": 0.95, "building": 0.75}
}

summary = {
    "total_pixels": 262144,
    "per_class_percentages": {"background": 49.6, "building": 19.1}
}

prompt = f"""You are an expert in remote sensing and land cover classification.

A semantic segmentation model has been applied to satellite/aerial imagery.

**Evaluation Metrics:**
{json.dumps(metrics, indent=2)}

**Segmentation Summary:**
{json.dumps(summary, indent=2)}

IMPORTANT: Return ONLY valid JSON, no markdown code blocks, no additional text.

Generate a report in JSON format with this structure:
{{
    "executive_summary": "A 3-4 sentence summary of key findings.",
    "urban_planning": ["Insight 1", "Insight 2"],
    "recommendations": {{"model_improvements": ["Suggestion 1"]}}
}}

Return ONLY the JSON object, nothing else."""

print("=" * 80)
print("Sending report prompt to Gemini API...")
print("=" * 80)

request_body = {
    "contents": [{
        "parts": [{"text": prompt}]
    }],
    "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 2048
    }
}

try:
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}",
        json=request_body,
        timeout=60.0
    )
    
    print(f"\nStatus Code: {response.status_code}\n")
    
    if response.status_code == 200:
        result = response.json()
        
        # Extract text
        try:
            text = result['candidates'][0]['content']['parts'][0]['text']
            print("Raw Response Text:")
            print("-" * 80)
            print(text)
            print("-" * 80)
            
            # Try to parse as JSON
            print("\nAttempting to parse as JSON...")
            try:
                report = json.loads(text)
                print("✓ Successfully parsed as JSON!")
                print(json.dumps(report, indent=2)[:500])
            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing failed: {e}")
                print(f"Error at position {e.pos}: {text[max(0, e.pos-50):e.pos+50]}")
                
        except (KeyError, IndexError, TypeError) as e:
            print(f"✗ Could not extract text from response: {e}")
            print(json.dumps(result, indent=2))
    else:
        print(f"✗ API returned status {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
