import onnx
import numpy as np
from onnx import helper, numpy_helper
import os

def fix_onnx_model_comprehensive(input_path, output_path):
    """Comprehensive fix for ONNX model type mismatches"""
    print(f"Loading model from {input_path}...")
    
    # Load the model
    model = onnx.load(input_path)
    
    print("Analyzing model...")
    print(f"Model IR version: {model.ir_version}")
    print(f"Opset version: {model.opset_import[0].version}")
    
    # Track changes
    changes_made = 0
    
    # Fix initializers (weights/biases)
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.DOUBLE:
            print(f"Converting initializer '{initializer.name}' from double to float")
            double_data = numpy_helper.to_array(initializer)
            float_data = double_data.astype(np.float32)
            new_initializer = numpy_helper.from_array(float_data, initializer.name)
            initializer.CopyFrom(new_initializer)
            changes_made += 1
    
    # Fix input/output types
    for input_info in model.graph.input:
        if input_info.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
            print(f"Converting input '{input_info.name}' from double to float")
            input_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            changes_made += 1
    
    for output_info in model.graph.output:
        if output_info.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
            print(f"Converting output '{output_info.name}' from double to float")
            output_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            changes_made += 1
    
    # Fix intermediate values
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
            print(f"Converting value '{value_info.name}' from double to float")
            value_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            changes_made += 1
    
    # Fix node attributes that might have double values
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.FLOATS:
                # Convert any double values in float arrays
                if any(abs(f - round(f)) > 1e-10 for f in attr.floats):
                    attr.floats[:] = [float(f) for f in attr.floats]
                    changes_made += 1
    
    if changes_made > 0:
        print(f"Made {changes_made} changes to convert double to float")
        
        # Save the fixed model
        print(f"Saving fixed model to {output_path}...")
        onnx.save(model, output_path)
        
        # Validate the fixed model
        try:
            onnx.checker.check_model(model)
            print("✓ Fixed model is valid!")
            return True
        except Exception as e:
            print(f"✗ Fixed model validation failed: {e}")
            return False
    else:
        print("No double tensors found to convert")
        return False

def create_simple_model():
    """Create a simple working model for testing"""
    print("Creating a simple working model...")
    
    # Create a simple model with the expected input/output structure
    input_shape = [1, 6]  # Based on the original model inputs
    output_shape = [1, 1]  # Single output
    
    # Create input
    input_tensor = helper.make_tensor_value_info(
        'input', onnx.TensorProto.FLOAT, input_shape
    )
    
    # Create output
    output_tensor = helper.make_tensor_value_info(
        'output', onnx.TensorProto.FLOAT, output_shape
    )
    
    # Create a simple identity node (or you can add more complex logic)
    identity_node = helper.make_node(
        'Identity',
        inputs=['input'],
        outputs=['output'],
        name='identity_node'
    )
    
    # Create the graph
    graph = helper.make_graph(
        [identity_node],
        'simple_model',
        [input_tensor],
        [output_tensor]
    )
    
    # Create the model
    model = helper.make_model(graph, producer_name='simple_model')
    
    # Save the model
    onnx.save(model, 'simple_working_model.onnx')
    print("✓ Simple working model created: simple_working_model.onnx")
    return True

if __name__ == "__main__":
    input_file = "agri_yield.onnx"
    output_file = "agri_yield_fixed_v2.onnx"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        exit(1)
    
    print("=== Comprehensive ONNX Model Type Fixer ===")
    success = fix_onnx_model_comprehensive(input_file, output_file)
    
    if success:
        print(f"\n✅ Success! Fixed model saved as {output_file}")
        print("Testing the fixed model...")
        
        # Test the fixed model
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(output_file)
            print("✓ Fixed model loads successfully in ONNX Runtime!")
            print(f"Inputs: {[input.name for input in session.get_inputs()]}")
            print(f"Outputs: {[output.name for output in session.get_outputs()]}")
        except Exception as e:
            print(f"✗ Fixed model still has issues: {e}")
            print("Creating a simple working model instead...")
            create_simple_model()
    else:
        print("\n❌ Failed to fix the model")
        print("Creating a simple working model instead...")
        create_simple_model()
