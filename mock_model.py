import numpy as np

class MockModel:
    """Mock model for testing when ONNX model fails to load"""
    
    def __init__(self):
        self.input_name = "input"
        self.output_name = "output"
    
    def get_inputs(self):
        return [MockInput()]
    
    def get_outputs(self):
        return [MockOutput()]
    
    def run(self, output_names, input_dict):
        # Simple mock prediction based on input features
        input_data = input_dict[self.input_name]
        # Mock agricultural yield prediction
        soil_ph = input_data[0][0]
        crop_type = input_data[0][1]
        season = input_data[0][2]
        region = input_data[0][3]
        
        # Simple mock logic
        base_yield = 0.5
        ph_factor = max(0, min(1, (soil_ph - 5.0) / 3.0))  # Optimal pH around 6.5
        crop_factor = 0.8 + (crop_type * 0.05)  # Different crops have different yields
        season_factor = 0.9 + (season * 0.1)    # Different seasons
        region_factor = 0.85 + (region * 0.1)   # Different regions
        
        prediction = base_yield * ph_factor * crop_factor * season_factor * region_factor
        prediction = max(0.1, min(1.0, prediction))  # Clamp between 0.1 and 1.0
        
        return [np.array([[prediction]], dtype=np.float32)]

class MockInput:
    def __init__(self):
        self.name = "input"

class MockOutput:
    def __init__(self):
        self.name = "output"
