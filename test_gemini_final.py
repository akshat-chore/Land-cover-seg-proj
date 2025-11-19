#!/usr/bin/env python3
"""
Final comprehensive test of the Gemini API report generation with the fix.
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
    "per_class_percentages": {
        "background": 49.6,
        "building": 19.1,
        "woodland": 15.3,
        "water": 9.5,
        "road": 6.5
    }
}

prompt = f"""You are an expert in remote sensing and land cover classification.

A semantic segmentation model has been applied to satellite/aerial imagery.

**Evaluation Metrics:**
{json.dumps(metrics, indent=2)}

**Segmentation Summary:**
{json.dumps(summary, indent=2)}

IMPORTANT: Return ONLY valid JSON wrapped in markdown code blocks, no additional text.

Generate a report in JSON format with this structure:
{{
    "executive_summary": "A 3-4 sentence summary of key findings.",
    "urban_planning": ["Insight 1", "Insight 2"],
    "disaster_management": ["Insight 1"],
    "automation_accuracy": ["Assessment"],
    "recommendations": {{"model_improvements": ["Suggestion"], "deployment_notes": ["Note"]}}
}}

Return ONLY the JSON object in markdown code blocks."""

print("=" * 80)
print("FINAL TEST: Gemini API Report Generation with JSON Extraction")
print("=" * 80)

request_body = {
    "contents": [{
        "parts": [{"text": prompt}]
    }],
    "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 3000
    }
}

try:
    print("\n1. Sending request to Gemini API...")
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={GEMINI_API_KEY}",
        json=request_body,
        timeout=60.0
    )
    
    print(f"   Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        # Extract text
        try:
            generated_text = result['candidates'][0]['content']['parts'][0]['text']
            print("   ✓ Response received successfully\n")
        except (KeyError, IndexError, TypeError) as e:
            print(f"   ✗ Could not extract text from response: {e}")
            print(f"   Response: {json.dumps(result, indent=2)[:500]}")
            raise
        
        # Apply the improved extraction logic
        print("2. Extracting JSON from markdown code blocks...")
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
        print("   ✓ Extracted text from code blocks\n")
        
        # Parse JSON
        print("3. Parsing extracted JSON...")
        try:
            report_json = json.loads(extracted_text)
            print("   ✓ Successfully parsed JSON!\n")
            
            # Display results
            print("4. Generated Report Structure:")
            print("   " + "=" * 76)
            for key, value in report_json.items():
                if isinstance(value, str):
                    preview = (value[:70] + "...") if len(value) > 70 else value
                    print(f"   • {key}: {preview}")
                elif isinstance(value, (list, dict)):
                    print(f"   • {key}: <{type(value).__name__}> with {len(value)} items")
                else:
                    print(f"   • {key}: <{type(value).__name__}>")
            print("   " + "=" * 76)
            
            print("\n" + "=" * 80)
            print("✓ SUCCESS! Gemini API is working correctly!")
            print("=" * 80)
            print("\nFull Report JSON (formatted):")
            print(json.dumps(report_json, indent=2))
            
        except json.JSONDecodeError as e:
            print(f"   ✗ JSON parsing failed: {e}")
            print(f"   Error at position {e.pos}")
            print(f"   Context: {extracted_text[max(0, e.pos-50):e.pos+50]}")
            raise
            
    else:
        print(f"   ✗ API returned status {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
