import onnx
import numpy as np
from onnx import helper, numpy_helper
import os

def fix_onnx_model(input_path, output_path):
    """Fix ONNX model by converting double tensors to float tensors"""
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
            # Convert double to float
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

if __name__ == "__main__":
    input_file = "agri_yield.onnx"
    output_file = "agri_yield_fixed.onnx"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        exit(1)
    
    print("=== ONNX Model Type Fixer ===")
    success = fix_onnx_model(input_file, output_file)
    
    if success:
        print(f"\n✅ Success! Fixed model saved as {output_file}")
        print("You can now use the fixed model in your FastAPI app.")
    else:
        print("\n❌ Failed to fix the model")
