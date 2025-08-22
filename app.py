from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import os
from mock_model import MockModel

app = FastAPI(title="ML Prediction API", description="FastAPI backend for ONNX model inference")

# Load model once
try:
    # Configure session options to handle type mismatches
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_options.enable_cpu_mem_arena = False
    
    # Load model with specific providers and disable optimizations
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession("agri_yield.onnx", session_options, providers=providers)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    # Try alternative approach with different ONNX Runtime version
    try:
        print("Trying alternative loading method...")
        session = ort.InferenceSession("agri_yield.onnx")
        print("Model loaded with basic method")
    except Exception as e2:
        print(f"Alternative loading also failed: {e2}")
        print("Using mock model for testing...")
        session = MockModel()

class InputData(BaseModel):
    features: list
    
    class Config:
        json_schema_extra = {
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
        "model_loaded": session is not None,
        "model_type": "mock" if isinstance(session, MockModel) else "onnx"
    }

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input list to numpy
        input_array = np.array(data.features, dtype=np.float32).reshape(1, -1)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_array})
        
        # Convert result to float32 to handle type mismatches
        prediction = result[0].astype(np.float32).tolist()
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
