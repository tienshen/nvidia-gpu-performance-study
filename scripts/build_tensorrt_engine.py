"""Build TensorRT engine from ONNX model."""
import argparse
import os
from pathlib import Path

try:
    import tensorrt as trt
except ImportError as exc:
    raise SystemExit("TensorRT python package not found. Install `tensorrt` and try again.") from exc


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path, engine_path, fp16=False, int8=False, max_workspace_size=2<<30):
    """Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save engine
        fp16: Enable FP16 precision
        int8: Enable INT8 precision
        max_workspace_size: Maximum workspace size in bytes (default: 2GB)
    """
    print(f"Building TensorRT engine from {onnx_path}")
    print(f"  FP16: {fp16}")
    print(f"  INT8: {int8}")
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")
    
    print(f"Network inputs: {network.num_inputs}")
    print(f"Network outputs: {network.num_outputs}")
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    
    if fp16:
        if not builder.platform_has_fast_fp16:
            print("Warning: Platform does not support fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
        print("Enabled FP16 precision")
    
    if int8:
        if not builder.platform_has_fast_int8:
            print("Warning: Platform does not support fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        print("Enabled INT8 precision")
    
    # Build engine
    print("Building engine (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine")
    
    # Save engine
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"Engine saved to: {engine_path}")
    print(f"Engine size: {engine_path.stat().st_size / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX model")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--output", default=None, help="Output engine path (default: auto-generated)")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 precision")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 precision")
    parser.add_argument("--workspace", type=int, default=2048, help="Max workspace size in MB (default: 2048)")
    args = parser.parse_args()
    
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"ONNX model not found: {onnx_path}")
    
    # Generate output path if not specified
    if args.output:
        engine_path = Path(args.output)
    else:
        # Create engine in engines/ directory with descriptive name
        engine_name = onnx_path.stem
        precision = []
        
        # Only add precision suffix if not already in ONNX model name
        if args.fp16 and "_fp16" not in engine_name:
            precision.append("fp16")
        if args.int8 and "_int8" not in engine_name:
            precision.append("int8")
        
        if precision:
            engine_name += f"_{'_'.join(precision)}"
        engine_name += ".plan"
        
        engine_path = Path("engines") / engine_name
    
    workspace_bytes = args.workspace * 1024 * 1024
    build_engine(
        str(onnx_path),
        str(engine_path),
        fp16=args.fp16,
        int8=args.int8,
        max_workspace_size=workspace_bytes
    )


if __name__ == "__main__":
    main()
