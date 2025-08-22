# FastAPI Backend with ONNX Model Inference

This is a FastAPI backend service that serves agricultural yield predictions using an ONNX model.

## Features

- Agricultural yield prediction based on soil pH, crop type, season, and region
- Fast inference using ONNX Runtime
- RESTful API with automatic documentation
- Health check endpoint

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The model `agri_yield.onnx` is already included and ready to use.

## Run Locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /predict

Send a POST request with JSON body:
```json
{
  "features": [6.5, 0, 1, 2]
}
```

Where the features represent:
- `soil_ph`: Soil pH value (float)
- `crop_name`: Crop type (encoded as integer)
- `season`: Season (encoded as integer) 
- `region`: Region (encoded as integer)

Response:
```json
{
  "prediction": [0.85]
}
```

The prediction represents the expected yield (0-1 scale).

## Deploy on Render

1. Connect your repository to Render
2. Create a new Web Service
3. Select Python as the runtime
4. Set the build command: `pip install -r requirements.txt`
5. Set the start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## Environment Variables

- `PORT`: Automatically set by Render (default: 8000)

## Model Information

The `agri_yield.onnx` model:
- Accepts 4 input features: soil_ph, crop_name, season, region
- Returns yield prediction as a single value (0-1 scale)
- Uses float32 data type
- Optimized for agricultural yield prediction
