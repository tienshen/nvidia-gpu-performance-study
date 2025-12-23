"""Script to quantize ONNX models to INT8."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.models import ensure_models_dir


def create_calibration_data_reader():
    """Create a calibration data reader for static quantization."""
    import numpy as np
    
    class CalibrationDataReader:
        def __init__(self, num_samples=20):
            self.num_samples = num_samples
            self.batch_size = 1
            self.seq_len = 128
            self.vocab_size = 30522
            self.current_sample = 0
            
        def get_next(self):
            if self.current_sample >= self.num_samples:
                return None
            
            # Generate random calibration data
            input_ids = np.random.randint(0, self.vocab_size, 
                                         size=(self.batch_size, self.seq_len), 
                                         dtype=np.int64)
            attention_mask = np.ones((self.batch_size, self.seq_len), dtype=np.int64)
            
            self.current_sample += 1
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
    
    return CalibrationDataReader()


def quantize_model(input_path: str, output_path: str, quantization_mode: str = "dynamic"):
    """
    Quantize an ONNX model to INT8.
    
    Args:
        input_path: Path to input FP32/FP16 ONNX model
        output_path: Path to save quantized model
        quantization_mode: 'dynamic' or 'static' quantization
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat
        import onnx
    except ImportError:
        print("Error: onnxruntime.quantization not available")
        print("Install with: pip install onnxruntime-gpu")
        sys.exit(1)
    
    print(f"Quantizing {input_path} to INT8...")
    print(f"Mode: {quantization_mode}")
    print(f"Output: {output_path}")
    
    input_model = Path(input_path)
    if not input_model.exists():
        print(f"Error: Input model not found: {input_path}")
        sys.exit(1)
    
    # Dynamic quantization (no calibration needed)
    if quantization_mode == "dynamic":
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
        )
    else:
        # Static quantization with calibration
        print("Generating calibration data (20 samples)...")
        calibration_data_reader = create_calibration_data_reader()
        
        # For TensorRT: symmetric activation and weight quantization
        extra_options = {
            'ActivationSymmetric': True,  # Symmetric activation quantization (zero point = 0)
            'WeightSymmetric': True,      # Symmetric weight quantization (default is True)
        }
        
        quantize_static(
            model_input=str(input_path),
            model_output=str(output_path),
            calibration_data_reader=calibration_data_reader,
            quant_format=QuantFormat.QDQ,  # QDQ format for better TensorRT compatibility
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            extra_options=extra_options,
        )
        print("Calibration complete")
    
    # Check output size
    output_model = Path(output_path)
    if output_model.exists():
        original_size = input_model.stat().st_size / (1024 * 1024)
        quantized_size = output_model.stat().st_size / (1024 * 1024)
        compression_ratio = original_size / quantized_size
        
        print(f"\n✓ Successfully quantized model")
        print(f"Original size: {original_size:.2f} MB")
        print(f"Quantized size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
    else:
        print("Error: Quantization failed")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ONNX models to INT8"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input ONNX model path (relative to models/ or absolute)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output quantized model path (default: input with _int8 suffix)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["dynamic", "static"],
        default="dynamic",
        help="Quantization mode (default: dynamic)"
    )
    
    args = parser.parse_args()
    
    # Ensure models directory exists
    models_dir = ensure_models_dir()
    
    # Resolve input path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = models_dir / args.input
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = models_dir / args.output
    else:
        # Auto-generate output name
        stem = input_path.stem
        # Replace precision suffix with int8
        if stem.endswith("_fp32"):
            stem = stem[:-5] + "_int8"
        elif stem.endswith("_fp16"):
            stem = stem[:-5] + "_int8"
        else:
            stem = stem + "_int8"
        output_path = models_dir / f"{stem}.onnx"
    
    quantize_model(str(input_path), str(output_path), args.mode)


if __name__ == "__main__":
    main()
