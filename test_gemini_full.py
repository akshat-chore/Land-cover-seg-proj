#!/usr/bin/env python3
"""
Test script to validate Gemini API integration for report generation.
"""

import asyncio
import json
import logging
from app.gemini_client import GeminiClient

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Example metrics
example_metrics = {
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

example_summary = {
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

example_context = {
    "region": "Downtown Manhattan",
    "date": "2024-09-15",
    "scenario": "Urban development assessment"
}


async def test_gemini_report():
    """Test the full report generation flow."""
    print("=" * 80)
    print("Testing Gemini API Report Generation")
    print("=" * 80)
    
    client = GeminiClient()
    
    try:
        print("\n1. Testing with valid metrics and context...")
        result = await client.generate_report(
            metrics_json=example_metrics,
            segmentation_summary=example_summary,
            context=example_context
        )
        
        print("\n✓ Report generation successful!")
        print(f"\nStatus: {result.get('status')}")
        
        if result.get('status') == 'success':
            report = result.get('report', {})
            print(f"\nReport keys: {list(report.keys())}")
            
            if 'executive_summary' in report:
                print(f"\nExecutive Summary:\n{report['executive_summary']}\n")
            
            if 'report_markdown' in report:
                print(f"\nGenerated Markdown Report (first 500 chars):\n{report['report_markdown'][:500]}...\n")
            
            print("\n✓ Full report structure:")
            print(json.dumps({k: f"<{type(v).__name__}>" if k != 'executive_summary' else v 
                            for k, v in report.items()}, indent=2))
        
        return result
        
    except Exception as e:
        print(f"\n✗ Error during report generation: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(test_gemini_report())
    
    if result and result.get('status') == 'success':
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED - Gemini API is working correctly!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ TESTS FAILED - Please check the errors above")
        print("=" * 80)
