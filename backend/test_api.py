import requests
import json

def test_predict():
    """Test the predict endpoint"""
    url = "http://localhost:8000/predict"
    
    # Test data for agricultural yield prediction
    data = {
        "features": [6.5, 0, 1, 2]  # soil_ph, crop_name, season, region
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_predict()
