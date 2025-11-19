#!/usr/bin/env python3
"""
Test the /report endpoint integration with the fixed Gemini client.
"""

import json
import asyncio
from app.gemini_client import generate_report_sync

# Example data
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

print("=" * 80)
print("ENDPOINT INTEGRATION TEST: /report")
print("=" * 80)

print("\nTesting generate_report_sync() function...")
print("(This is what the /report endpoint calls)")
print()

try:
    result = generate_report_sync(metrics, summary, context)
    
    if result.get('status') == 'success':
        print("✓ Report generation successful!\n")
        
        report = result.get('report', {})
        print("Generated Report Structure:")
        print("-" * 80)
        for key in report.keys():
            value = report[key]
            if isinstance(value, str):
                preview = (value[:60] + "...") if len(value) > 60 else value
                print(f"  ✓ {key}: {preview}")
            elif isinstance(value, list):
                print(f"  ✓ {key}: [{len(value)} items]")
            elif isinstance(value, dict):
                print(f"  ✓ {key}: {{...}} with {len(value)} keys")
            else:
                print(f"  ✓ {key}: <{type(value).__name__}>")
        print("-" * 80)
        
        print("\n" + "=" * 80)
        print("✓ INTEGRATION TEST PASSED")
        print("=" * 80)
        print("\nThe /report endpoint is now fully functional!")
        print("\nExample API Response (summary):")
        print(json.dumps({
            "status": "success",
            "report_keys": list(report.keys()),
            "sample_summary": report.get("executive_summary", "")[:100] + "..."
        }, indent=2))
        
    else:
        print(f"✗ Report generation failed: {result.get('message')}")
        
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
