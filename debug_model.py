import onnxruntime as ort
import numpy as np

print("Testing ONNX model loading...")

try:
    # Method 1: Basic loading
    print("Method 1: Basic loading")
    session1 = ort.InferenceSession("agri_yield.onnx")
    print("✓ Basic loading successful")
except Exception as e:
    print(f"✗ Basic loading failed: {e}")

try:
    # Method 2: With session options
    print("\nMethod 2: With session options")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session2 = ort.InferenceSession("agri_yield.onnx", session_options)
    print("✓ Session options loading successful")
except Exception as e:
    print(f"✗ Session options loading failed: {e}")

try:
    # Method 3: With specific providers
    print("\nMethod 3: With specific providers")
    providers = ['CPUExecutionProvider']
    session3 = ort.InferenceSession("agri_yield.onnx", providers=providers)
    print("✓ Provider loading successful")
except Exception as e:
    print(f"✗ Provider loading failed: {e}")

try:
    # Method 4: Combined approach
    print("\nMethod 4: Combined approach")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ['CPUExecutionProvider']
    session4 = ort.InferenceSession("agri_yield.onnx", session_options, providers=providers)
    print("✓ Combined loading successful")
    
    # Test inference
    print("\nTesting inference...")
    input_data = np.array([6.5, 0, 1, 2], dtype=np.float32).reshape(1, -1)
    input_name = session4.get_inputs()[0].name
    output_name = session4.get_outputs()[0].name
    result = session4.run([output_name], {input_name: input_data})
    print(f"✓ Inference successful: {result[0]}")
    
except Exception as e:
    print(f"✗ Combined loading failed: {e}")
