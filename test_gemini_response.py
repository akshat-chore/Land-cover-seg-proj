"""
Test script to check Gemini API response structure
"""
import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

if not api_key:
    print("ERROR: GEMINI_API_KEY not set")
    exit(1)

print(f"Testing Gemini API with model: {model_name}")

# Simple test request
request_body = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Say hello in one word"
                }
            ]
        }
    ],
    "generationConfig": {
        "temperature": 0.7,
        "topP": 0.9,
        "topK": 40,
        "maxOutputTokens": 100
    }
}

try:
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}",
        json=request_body,
        timeout=60.0
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"\nFull Response:")
    print(json.dumps(response.json(), indent=2))
    
    # Try to extract text
    result = response.json()
    print("\n\nAttempting to extract text...")
    
    try:
        text = result['candidates'][0]['content']['parts'][0]['text']
        print(f"SUCCESS - Extracted text: {text}")
    except (KeyError, IndexError, TypeError) as e:
        print(f"ERROR - Could not extract text: {e}")
        print(f"Response keys: {result.keys()}")
        if 'candidates' in result:
            print(f"Candidates: {result['candidates']}")
            if result['candidates']:
                print(f"First candidate keys: {result['candidates'][0].keys()}")
                if 'content' in result['candidates'][0]:
                    print(f"Content keys: {result['candidates'][0]['content'].keys()}")
                    print(f"Content value: {result['candidates'][0]['content']}")
    
except Exception as e:
    print(f"ERROR: {e}")
