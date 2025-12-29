"""Build TensorRT engine from ONNX model."""
import argparse
import subprocess
from pathlib import Path

try:
    import tensorrt as trt
except ImportError as exc:
    raise SystemExit("TensorRT python package not found. Install `tensorrt` and try again.") from exc


#TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def _build_engine_with_trtexec(
    onnx_path,
    engine_path,
    fp16=False,
    int8=False,
    max_workspace_size=2 << 30,
    trtexec_path="trtexec",
    min_shapes=None,
    opt_shapes=None,
    max_shapes=None,
    calib_cache=None,
    load_inputs=None,
    dump_layer_info=False,
    export_layer_info=None,
    profiling_verbosity=None,
):
    workspace_mb = max(1, int(max_workspace_size // (1024 * 1024)))
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}M",
        # --explicitBatch is deprecated and now default for ONNX
    ]
    if fp16:
        cmd.append("--fp16")
    if int8:
        cmd.append("--int8")
    if min_shapes:
        cmd.append(f"--minShapes={min_shapes}")
    if opt_shapes:
        cmd.append(f"--optShapes={opt_shapes}")
    if max_shapes:
        cmd.append(f"--maxShapes={max_shapes}")
    if calib_cache:
        cmd.append(f"--calib={calib_cache}")
    if load_inputs:
        cmd.append(f"--loadInputs={load_inputs}")
    if dump_layer_info:
        cmd.append("--dumpLayerInfo")
    if export_layer_info:
        cmd.append(f"--exportLayerInfo={export_layer_info}")
    if profiling_verbosity:
        cmd.append(f"--profilingVerbosity={profiling_verbosity}")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_engine(
    onnx_path,
    engine_path,
    fp16=False,
    int8=False,
    max_workspace_size=2<<30,
    calib_samples=100,
    trtexec_path="trtexec",
    min_shapes=None,
    opt_shapes=None,
    max_shapes=None,
    calib_cache=None,
    load_inputs=None,
    dump_layer_info=False,
    export_layer_info=None,
    profiling_verbosity=None,
):
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
    
    if int8:
        print("Using trtexec for INT8 build; Python calibrator is disabled.")
        _build_engine_with_trtexec(
            onnx_path=onnx_path,
            engine_path=engine_path,
            fp16=fp16,
            int8=int8,
            max_workspace_size=max_workspace_size,
            trtexec_path=trtexec_path,
            min_shapes=min_shapes,
            opt_shapes=opt_shapes,
            max_shapes=max_shapes,
            calib_cache=calib_cache,
            load_inputs=load_inputs,
            dump_layer_info=dump_layer_info,
            export_layer_info=export_layer_info,
            profiling_verbosity=profiling_verbosity,
        )
        return

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
        raise RuntimeError("INT8 build uses trtexec; rerun with --int8 to trigger it.")
    
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
    parser.add_argument("--calib-samples", type=int, default=100, help="Number of random calibration samples for INT8 (ignored when using trtexec)")
    parser.add_argument("--trtexec", default="trtexec", help="Path to trtexec executable (default: trtexec)")
    parser.add_argument("--min-shapes", default=None, help="minShapes for trtexec, e.g. input_ids:1x128,attention_mask:1x128")
    parser.add_argument("--opt-shapes", default=None, help="optShapes for trtexec, e.g. input_ids:1x128,attention_mask:1x128")
    parser.add_argument("--max-shapes", default=None, help="maxShapes for trtexec, e.g. input_ids:1x128,attention_mask:1x128")
    parser.add_argument("--calib-cache", default=None, help="Calibration cache path for trtexec INT8 builds")
    parser.add_argument("--load-inputs", default=None, help="trtexec --loadInputs string for calibration/inference data")
    parser.add_argument("--dump-layer-info", action="store_true", help="Pass --dumpLayerInfo to trtexec")
    parser.add_argument("--export-layer-info", default=None, help="Pass --exportLayerInfo=<path> to trtexec")
    parser.add_argument("--profiling-verbosity", default=None, help="Pass --profilingVerbosity=<level> to trtexec")
    args = parser.parse_args()
    
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise SystemExit(f"ONNX model not found: {onnx_path}")
    
    # Generate output path if not specified
    if args.output:
        engine_path = Path(args.output)
        engine_name = engine_path.stem
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

    export_layer_info = args.export_layer_info
    if args.dump_layer_info and not export_layer_info:
        export_layer_info = str(Path("engines") / f"{engine_name}_layer_info.json")
    
    workspace_bytes = args.workspace * 1024 * 1024
    build_engine(
        str(onnx_path),
        str(engine_path),
        fp16=args.fp16,
        int8=args.int8,
        max_workspace_size=workspace_bytes,
        calib_samples=args.calib_samples,
        trtexec_path=args.trtexec,
        min_shapes=args.min_shapes,
        opt_shapes=args.opt_shapes,
        max_shapes=args.max_shapes,
        calib_cache=args.calib_cache,
        load_inputs=args.load_inputs,
        dump_layer_info=args.dump_layer_info,
        export_layer_info=export_layer_info,
        profiling_verbosity=args.profiling_verbosity,
    )


if __name__ == "__main__":
    main()
