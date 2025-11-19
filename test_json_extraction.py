#!/usr/bin/env python3
"""
Test the improved JSON extraction from Gemini markdown code blocks.
"""

import json

# Simulating the Gemini response with markdown code blocks
gemini_response = '''```json
{
  "executive_summary": "The semantic segmentation model demonstrates strong overall performance with a high pixel accuracy of 0.92 and a mean IoU of 0.78. While the 'background' class is segmented with excellent precision (0.95 IoU), the 'building' class, representing 19.1% of the total pixels, shows good but comparatively lower accuracy (0.75 IoU). This indicates a robust capability for general land cover mapping, with specific room for improvement in building delineation.",
  "urban_planning": [
    "The model provides valuable quantitative data on building distribution and density (19.1% coverage), enabling urban planners to monitor urban sprawl, assess land use patterns, and identify areas for potential development or redevelopment.",
    "Accurate building footprints can significantly enhance infrastructure planning, aid in zoning compliance monitoring, and support disaster preparedness and response efforts by providing precise locations of built structures."
  ],
  "recommendations": {
    "model_improvements": [
      "Prioritize improving the segmentation accuracy for the 'building' class, potentially by incorporating more diverse training data (e.g., varying building types, scales, and environmental conditions), experimenting with different loss functions, or refining model architecture to better capture complex building geometries and boundaries."
    ]
  }
}
```'''

print("Testing JSON extraction from markdown code blocks...")
print("=" * 80)

# Apply the same logic from the fixed code
extracted_text = gemini_response.strip()

# Remove markdown code block markers if present
if extracted_text.startswith("```json"):
    extracted_text = extracted_text[7:]  # Remove ```json
    if extracted_text.endswith("```"):
        extracted_text = extracted_text[:-3]  # Remove ```
elif extracted_text.startswith("```"):
    extracted_text = extracted_text[3:]  # Remove ```
    if extracted_text.endswith("```"):
        extracted_text = extracted_text[:-3]  # Remove ```

extracted_text = extracted_text.strip()

print("Extracted text (first 200 chars):")
print(extracted_text[:200])
print()

try:
    report_json = json.loads(extracted_text)
    print("✓ Successfully parsed JSON!")
    print("\nJSON structure:")
    print(json.dumps({k: f"<{type(v).__name__}>" for k, v in report_json.items()}, indent=2))
    print("\nSample content:")
    print(json.dumps({
        "executive_summary": report_json.get("executive_summary", "")[:100] + "...",
        "urban_planning_count": len(report_json.get("urban_planning", [])),
        "has_recommendations": "recommendations" in report_json
    }, indent=2))
except json.JSONDecodeError as e:
    print(f"✗ Failed to parse JSON: {e}")
    print(f"Error at position {e.pos}")
