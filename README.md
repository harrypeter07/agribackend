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
  "rainfall": 100.0,
  "temperature": 25.0,
  "humidity": 70.0,
  "soil_ph": 6.5,
  "fertilizer_usage": 50.0,
  "risk_score": 0.3
}
```

Where the inputs represent:
- `rainfall`: Rainfall in mm (float)
- `temperature`: Temperature in Celsius (float)
- `humidity`: Humidity percentage (float)
- `soil_ph`: Soil pH value (float)
- `fertilizer_usage`: Fertilizer usage in kg/ha (float)
- `risk_score`: Risk score (0-1 scale, float)

Response:
```json
{
  "prediction": [[0.85]]
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

The `working_agricultural_model.onnx` model:
- Accepts 6 input features: rainfall, temperature, humidity, soil_ph, fertilizer_usage, risk_score
- Returns yield prediction as a single value (0-1 scale)
- Uses float32 data type
- Optimized for agricultural yield prediction
