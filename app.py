from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import os
from mock_model import MockModel

app = FastAPI(title="ML Prediction API", description="FastAPI backend for ONNX model inference")

# Load model once
session = None
try:
    # Simple model loading without complex options
    session = ort.InferenceSession("working_agricultural_model.onnx")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    try:
        print("Using mock model for testing...")
        session = MockModel()
    except Exception as e2:
        print(f"Mock model also failed: {e2}")
        session = None

class InputData(BaseModel):
    rainfall: float
    temperature: float
    humidity: float
    soil_ph: float
    fertilizer_usage: float
    risk_score: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "rainfall": 100.0,
                "temperature": 25.0,
                "humidity": 70.0,
                "soil_ph": 6.5,
                "fertilizer_usage": 50.0,
                "risk_score": 0.3
            }
        }

@app.get("/")
def read_root():
    return {"message": "ML Prediction API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": session is not None,
        "model_type": "mock" if isinstance(session, MockModel) else "onnx"
    }

@app.post("/predict")
def predict(data: InputData):
    try:
        # Prepare input data for the model
        input_data = {
            'rainfall': np.array([[data.rainfall]], dtype=np.float32),
            'temperature': np.array([[data.temperature]], dtype=np.float32),
            'humidity': np.array([[data.humidity]], dtype=np.float32),
            'soil_ph': np.array([[data.soil_ph]], dtype=np.float32),
            'fertilizer_usage': np.array([[data.fertilizer_usage]], dtype=np.float32),
            'risk_score': np.array([[data.risk_score]], dtype=np.float32)
        }
        
        # Run inference
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], input_data)
        
        # Convert result to float32
        prediction = result[0].astype(np.float32).tolist()
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
