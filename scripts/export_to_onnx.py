"""Script to export HuggingFace models to ONNX format."""

import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.models import ModelLoader, ensure_models_dir


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace models to ONNX format"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="HuggingFace model name (e.g., 'bert-base-uncased', 'distilbert-base-uncased')"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: model_name.onnx)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache HuggingFace models"
    )
    parser.add_argument(
        "--batch", 
        type=int,
        default=1,
        help="Batch size for export (default: 1)"
    )
    parser.add_argument(
        "--static-shapes",
        action="store_true",
        help="Export with fixed batch/seq shapes (recommended for CoreML)"
    )
    parser.add_argument(
        "--fast-gelu",
        action="store_true",
        help="Replace GELU with tanh approximation before export to avoid Erf"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export model in FP16 precision"
    )

    
    args = parser.parse_args()
    
    # Determine output name
    suffix = ""
    if args.static_shapes:
        suffix += f"_b{args.batch}_s{args.max_length}"
    
    # Add GELU variant suffix
    if args.fast_gelu:
        suffix += "_fast-gelu"
    else:
        suffix += "_gelu"
    
    # Add precision suffix (always explicit: fp32 or fp16)
    if args.fp16:
        suffix += "_fp16"
    else:
        suffix += "_fp32"

    if args.output_name:
        output_name = args.output_name
    else:
        output_name = args.model_name.replace("/", "_") + suffix
    
    if not output_name.endswith(".onnx"):
        output_name += ".onnx"
    
    # Ensure models directory exists
    models_dir = ensure_models_dir()
    output_path = models_dir / output_name
    
    print(f"Exporting {args.model_name} to ONNX...")
    print(f"Output path: {output_path}")
    
    try:
        # Load model
        loader = ModelLoader(args.model_name, cache_dir=args.cache_dir)
        loader.load_from_huggingface()

        # Apply fast GELU if requested
        if args.fast_gelu:
            loader.apply_fast_gelu()
        
        # Convert to FP16 if requested
        if args.fp16:
            print("Converting model to FP16...")
            loader.model = loader.model.half()
        
        # # Create sample input
        # sample_input = loader.create_sample_input(max_length=args.max_length)

        # # Force batch size by repeating along axis 0 for each tensor
        # if args.batch != 1:
        #     for k, v in sample_input.items():
        #         # v is numpy array, shape [1, seq] usually
        #         sample_input[k] = v.repeat(args.batch, axis=0)

        # Create sample input
        text = ["hello world"] * args.batch
        sample_input = loader.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        
        # Convert inputs to FP16 if model is FP16
        if args.fp16:
            sample_input = {k: v.half() if v.dtype == torch.float32 else v 
                          for k, v in sample_input.items()}
        
        # Export to ONNX
        loader.export_to_onnx(
            str(output_path),
            input_sample=sample_input,
            opset_version=args.opset_version,
            static_shapes=args.static_shapes,
        )

        # Post-process: Convert ONNX to FP16 if requested
        # This is more reliable than exporting FP16 directly
        if args.fp16:
            print("\nConverting ONNX model to FP16...")
            try:
                from onnxconverter_common import float16
                import onnx
                
                model_onnx = onnx.load(str(output_path))
                model_fp16 = float16.convert_float_to_float16(model_onnx)
                onnx.save(model_fp16, str(output_path))
                print("✓ ONNX model converted to FP16")
            except ImportError:
                print("⚠ onnxconverter-common not installed. Install with: pip install onnxconverter-common")
                print("  Continuing with PyTorch-exported FP16 (may have mixed precision)")
        
        print(f"\n✓ Successfully exported to {output_path}")
        print(f"Model size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"\n✗ Error exporting model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
