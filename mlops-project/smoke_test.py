#!/usr/bin/env python3
"""Smoke test script for the Iris Classification API."""

import requests
import json
import sys
from typing import Dict, Any

# Configuration
API_URL = "http://localhost:8000"  # Change this to your deployed API URL


def test_predict() -> bool:
    """Test the /predict endpoint."""
    print("Testing /predict endpoint...")
    
    url = f"{API_URL}/predict"
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Prediction successful!")
        print(f"  Response: {json.dumps(data, indent=2)}")
        
        # Validate response structure
        if "data" in data:
            result = data["data"]
            assert "label" in result, "Missing 'label' in response"
            assert "label_id" in result, "Missing 'label_id' in response"
            assert "probability" in result, "Missing 'probability' in response"
            print(f"  ✓ Response structure valid")
            print(f"  ✓ Predicted: {result['label']} (ID: {result['label_id']}) with {result['probability']*100:.2f}% confidence")
        else:
            # Handle direct response format
            assert "label" in data or "prediction" in data, "Invalid response format"
            print(f"  ✓ Response received (legacy format)")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection error: Could not connect to {API_URL}")
        print(f"  Make sure the API is running and the URL is correct.")
        return False
    except requests.exceptions.Timeout:
        print(f"✗ Timeout: Request took too long")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP error: {e}")
        if hasattr(e.response, 'text'):
            print(f"  Response: {e.response.text}")
        return False
    except AssertionError as e:
        print(f"✗ Validation error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_health() -> bool:
    """Test the /health endpoint."""
    print("\nTesting /health endpoint...")
    
    url = f"{API_URL}/health"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Health check successful!")
        print(f"  Response: {json.dumps(data, indent=2)}")
        return True
        
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_root() -> bool:
    """Test the root endpoint."""
    print("\nTesting / endpoint...")
    
    url = f"{API_URL}/"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        print(f"✓ Root endpoint successful!")
        print(f"  Response: {json.dumps(data, indent=2)}")
        return True
        
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Iris Classification API - Smoke Test")
    print("=" * 60)
    print(f"API URL: {API_URL}\n")
    
    results = []
    
    # Test endpoints
    results.append(("Health Check", test_health()))
    results.append(("Root Endpoint", test_root()))
    results.append(("Predict Endpoint", test_predict()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed. ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()

