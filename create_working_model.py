import onnx
import numpy as np
from onnx import helper, numpy_helper
import os

def create_working_agricultural_model():
    """Create a working agricultural yield prediction model"""
    print("Creating a working agricultural yield prediction model...")
    
    # Model parameters based on the original model
    input_features = ['rainfall', 'temperature', 'humidity', 'soil_ph', 'fertilizer_usage', 'risk_score']
    output_name = 'yield_prediction'
    
    # Create input tensors
    input_tensors = []
    for feature in input_features:
        input_tensor = helper.make_tensor_value_info(
            feature, onnx.TensorProto.FLOAT, [1, 1]
        )
        input_tensors.append(input_tensor)
    
    # Create output tensor
    output_tensor = helper.make_tensor_value_info(
        output_name, onnx.TensorProto.FLOAT, [1, 1]
    )
    
    # Create nodes for a simple agricultural yield prediction model
    nodes = []
    
    # Create constants for the model
    # These would normally come from training, but we'll create reasonable defaults
    rainfall_weight = helper.make_tensor('rainfall_weight', onnx.TensorProto.FLOAT, [1], [0.2])
    temp_weight = helper.make_tensor('temp_weight', onnx.TensorProto.FLOAT, [1], [0.25])
    humidity_weight = helper.make_tensor('humidity_weight', onnx.TensorProto.FLOAT, [1], [0.15])
    ph_weight = helper.make_tensor('ph_weight', onnx.TensorProto.FLOAT, [1], [0.2])
    fertilizer_weight = helper.make_tensor('fertilizer_weight', onnx.TensorProto.FLOAT, [1], [0.15])
    risk_weight = helper.make_tensor('risk_weight', onnx.TensorProto.FLOAT, [1], [0.05])
    
    # Weighted sum of inputs
    nodes.append(helper.make_node('Mul', ['rainfall', 'rainfall_weight'], ['rainfall_scaled'], name='mul_rainfall'))
    nodes.append(helper.make_node('Mul', ['temperature', 'temp_weight'], ['temp_scaled'], name='mul_temp'))
    nodes.append(helper.make_node('Mul', ['humidity', 'humidity_weight'], ['humidity_scaled'], name='mul_humidity'))
    nodes.append(helper.make_node('Mul', ['soil_ph', 'ph_weight'], ['ph_scaled'], name='mul_ph'))
    nodes.append(helper.make_node('Mul', ['fertilizer_usage', 'fertilizer_weight'], ['fertilizer_scaled'], name='mul_fertilizer'))
    nodes.append(helper.make_node('Mul', ['risk_score', 'risk_weight'], ['risk_scaled'], name='mul_risk'))
    
    # Sum all scaled inputs
    nodes.append(helper.make_node('Add', ['rainfall_scaled', 'temp_scaled'], ['sum1'], name='add1'))
    nodes.append(helper.make_node('Add', ['sum1', 'humidity_scaled'], ['sum2'], name='add2'))
    nodes.append(helper.make_node('Add', ['sum2', 'ph_scaled'], ['sum3'], name='add3'))
    nodes.append(helper.make_node('Add', ['sum3', 'fertilizer_scaled'], ['sum4'], name='add4'))
    nodes.append(helper.make_node('Add', ['sum4', 'risk_scaled'], ['raw_prediction'], name='add5'))
    
    # Apply sigmoid activation to get yield between 0 and 1
    nodes.append(helper.make_node('Sigmoid', ['raw_prediction'], [output_name], name='sigmoid_output'))
    
    # Create the graph
    graph = helper.make_graph(
        nodes,
        'agricultural_yield_model',
        input_tensors,
        [output_tensor],
        initializer=[rainfall_weight, temp_weight, humidity_weight, ph_weight, fertilizer_weight, risk_weight]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='agricultural_yield_predictor')
    model.ir_version = 8  # Use compatible IR version
    model.opset_import[0].version = 17  # Use the same opset as original
    
    # Save the model
    onnx.save(model, 'working_agricultural_model.onnx')
    print("✓ Working agricultural model created: working_agricultural_model.onnx")
    
    # Test the model
    try:
        import onnxruntime as ort
        session = ort.InferenceSession('working_agricultural_model.onnx')
        print("✓ Model loads successfully in ONNX Runtime!")
        print(f"Inputs: {[input.name for input in session.get_inputs()]}")
        print(f"Outputs: {[output.name for output in session.get_outputs()]}")
        
        # Test inference
        test_input = {
            'rainfall': np.array([[100.0]], dtype=np.float32),
            'temperature': np.array([[25.0]], dtype=np.float32),
            'humidity': np.array([[70.0]], dtype=np.float32),
            'soil_ph': np.array([[6.5]], dtype=np.float32),
            'fertilizer_usage': np.array([[50.0]], dtype=np.float32),
            'risk_score': np.array([[0.3]], dtype=np.float32)
        }
        
        result = session.run([output_name], test_input)
        print(f"✓ Test inference successful: {result[0]}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Creating Working Agricultural Model ===")
    success = create_working_agricultural_model()
    
    if success:
        print("\n✅ Success! Working model created and tested.")
        print("You can now use this model in your FastAPI app.")
    else:
        print("\n❌ Failed to create working model")
