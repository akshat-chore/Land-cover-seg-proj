"""
Gemini API client for generating intelligent reports from segmentation metrics.
Uses async HTTP requests to call Google's Generative AI API.
"""

import os
import json
import logging
import httpx
import asyncio
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GeminiClient:
    """Client for Gemini API integration."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key (Optional[str]): API key. If None, reads from GEMINI_API_KEY env var.
            model_name (Optional[str]): Model name. If None, uses GEMINI_MODEL env var or defaults to 'gemini-pro'.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model_name or os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found in environment variables")
    
    def _build_report_prompt(self, metrics_json: Dict, segmentation_summary: Dict, 
                            context: Optional[Dict] = None) -> str:
        """
        Build a comprehensive prompt for Gemini.
        
        Args:
            metrics_json (Dict): Evaluation metrics from model.
            segmentation_summary (Dict): Summary of segmentation results (areas, percentages).
            context (Optional[Dict]): Additional context (region, date, scenario).
        
        Returns:
            str: Formatted prompt for Gemini.
        
        TODO: Customize the prompt based on application domain (urban planning, disaster management, etc.)
        """
        timestamp = datetime.now().isoformat()
        
        prompt = f"""You are an expert in remote sensing, land cover classification, and geospatial analysis.

A semantic segmentation model has been applied to satellite/aerial imagery. Below are the results:

**Evaluation Metrics (Model Performance):**
{json.dumps(metrics_json, indent=2)}

**Segmentation Summary (Areas & Percentages):**
{json.dumps(segmentation_summary, indent=2)}

**Context Information:**
{json.dumps(context, indent=2) if context else "No context provided."}

**Timestamp:** {timestamp}

Generate a detailed report in JSON format with the following structure (all fields required):
{{
    "executive_summary": "A 3-4 sentence summary of key findings",
    "urban_planning": [
        "Insight 1 relevant to urban planning",
        "Insight 2 relevant to urban planning"
    ],
    "disaster_management": [
        "Insight 1 relevant to disaster management",
        "Insight 2 relevant to disaster management"
    ],
    "automation_accuracy": [
        "Assessment of model automation readiness",
        "Recommendations for production deployment"
    ],
    "recommendations": {{
        "model_improvements": [
            "Improvement suggestion 1",
            "Improvement suggestion 2"
        ],
        "deployment_notes": [
            "Deployment note 1",
            "Deployment note 2"
        ]
    }}
}}

CRITICAL INSTRUCTIONS:
1. Return ONLY valid JSON wrapped in markdown code blocks (```json ... ```)
2. Keep all string values as single lines - no actual newlines in JSON
3. Escape all special characters properly with backslashes
4. Use only double quotes for JSON strings
5. Do NOT include any fields beyond what is specified above
6. No additional text before or after the code block

Return ONLY the properly formatted JSON."""
        
        return prompt
    
    async def generate_report(self, metrics_json: Dict, segmentation_summary: Dict, 
                            context: Optional[Dict] = None) -> Dict:
        """
        Generate a comprehensive report using Gemini API.
        
        Args:
            metrics_json (Dict): Evaluation metrics.
            segmentation_summary (Dict): Segmentation summary.
            context (Optional[Dict]): Context information.
        
        Returns:
            Dict: Report JSON with structured outputs and markdown.
        
        Raises:
            ValueError: If API key is not set.
            httpx.HTTPError: If API call fails.
        """
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        prompt = self._build_report_prompt(metrics_json, segmentation_summary, context)
        
        logger.info(f"Sending request to Gemini API (model: {self.model_name})")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        request_body = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "topK": 40,
                "maxOutputTokens": 4096
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_url}/{self.model_name}:generateContent?key={self.api_key}",
                    json=request_body
                )
                response.raise_for_status()
                
                result = response.json()
                logger.info("Gemini API response received successfully")
                logger.debug(f"API response structure: {json.dumps(result, indent=2)[:500]}")
                
                # Extract the generated text - handle different response structures
                try:
                    generated_text = result['candidates'][0]['content']['parts'][0]['text']
                except (KeyError, IndexError, TypeError) as e:
                    logger.error(f"Response structure: {result}")
                    raise KeyError(f"Could not extract text from response. Response keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}") from e
                
                # Parse JSON from generated text
                # Gemini often wraps JSON in markdown code blocks
                extracted_text = generated_text.strip()
                
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
                
                try:
                    report_json = json.loads(extracted_text)
                except json.JSONDecodeError as parse_error:
                    logger.error(f"Failed to parse extracted text as JSON: {parse_error}")
                    logger.debug(f"Extracted text (first 500 chars): {extracted_text[:500]}")
                    logger.error(f"Error at position {parse_error.pos} (line {parse_error.lineno}, col {parse_error.colno})")
                    # Show context
                    try:
                        lines = extracted_text.split('\n')
                        if parse_error.lineno <= len(lines):
                            error_line = lines[parse_error.lineno - 1]
                            logger.error(f"Error context: {error_line[max(0, parse_error.colno-50):parse_error.colno+50]}")
                    except:
                        pass
                    raise ValueError(f"Response text is not valid JSON. Parse error at line {parse_error.lineno}, col {parse_error.colno}: {parse_error.msg}") from parse_error
                
                return {
                    "status": "success",
                    "report": report_json,
                    "raw_prompt": prompt,
                    "raw_response": generated_text
                }
        
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            logger.error(f"Gemini API HTTP error: {status_code} - {e.response.text}")
            if status_code == 401:
                raise ValueError("Invalid GEMINI_API_KEY") from e
            elif status_code == 429:
                raise ValueError("Gemini API rate limit exceeded") from e
            else:
                raise
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing Gemini API response: {e}")
            logger.error(f"Make sure response contains: candidates[0].content.parts[0].text")
            raise
        except ValueError as e:
            logger.error(f"Invalid response format from Gemini: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling Gemini API: {type(e).__name__}: {e}")
            raise


# Sync wrapper for use in FastAPI endpoints (FastAPI can handle this in thread pool)
def generate_report_sync(metrics_json: Dict, segmentation_summary: Dict, 
                        context: Optional[Dict] = None) -> Dict:
    """
    Synchronous wrapper for generate_report.
    Use this for FastAPI endpoints that cannot be async.
    
    Args:
        metrics_json (Dict): Evaluation metrics.
        segmentation_summary (Dict): Segmentation summary.
        context (Optional[Dict]): Context information.
    
    Returns:
        Dict: Report JSON with structured outputs and markdown.
    """
    import nest_asyncio
    
    # Allow nested asyncio calls
    try:
        nest_asyncio.apply()
    except RuntimeError:
        pass  # Already applied
    
    client = GeminiClient()
    try:
        report = asyncio.run(client.generate_report(metrics_json, segmentation_summary, context))
        return report
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # We're in an event loop, create a new thread
            import concurrent.futures
            import threading
            
            result = {}
            def run_in_thread():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    report = loop.run_until_complete(
                        client.generate_report(metrics_json, segmentation_summary, context)
                    )
                    result['report'] = report
                finally:
                    loop.close()
            
            thread = threading.Thread(target=run_in_thread)
            thread.start()
            thread.join()
            return result.get('report', {
                "status": "error",
                "message": "Failed to generate report in thread",
                "report": None,
                "raw_prompt": None,
                "raw_response": None
            })
        else:
            logger.error(f"Error generating report: {e}")
            return {
                "status": "error",
                "message": str(e),
                "report": None,
                "raw_prompt": None,
                "raw_response": None
            }
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return {
            "status": "error",
            "message": str(e),
            "report": None,
            "raw_prompt": None,
            "raw_response": None
        }


# Example usage for testing
if __name__ == "__main__":
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
        "scenario": "Urban development assessment",
        "image_resolution_m": 0.5
    }
    
    print("Example prompt and metrics:")
    print("=" * 80)
    print(f"Metrics: {json.dumps(example_metrics, indent=2)}")
    print(f"Summary: {json.dumps(example_summary, indent=2)}")
    print(f"Context: {json.dumps(example_context, indent=2)}")
