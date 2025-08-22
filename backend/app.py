from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import os

app = FastAPI(title="ML Prediction API", description="FastAPI backend for ONNX model inference")

# Load model once
try:
    session = ort.InferenceSession("agri_yield.onnx")
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    session = None

class InputData(BaseModel):
    features: list
    
    class Config:
        schema_extra = {
            "example": {
                "features": [6.5, 0, 1, 2]  # soil_ph, crop_name, season, region (encoded)
            }
        }

@app.get("/")
def read_root():
    return {"message": "ML Prediction API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": session is not None
    }

@app.post("/predict")
def predict(data: InputData):
    if session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input list to numpy
        input_array = np.array(data.features, dtype=np.float32).reshape(1, -1)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_array})
        
        return {"prediction": result[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
