import argparse
import json
import os
import socket
import time

import numpy as np

try:
    import tensorrt as trt
except ImportError as exc:
    raise SystemExit("TensorRT python package not found. Install `tensorrt` and try again.") from exc

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError as exc:
    raise SystemExit("pycuda not found. Install `pycuda` and try again.") from exc


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def _parse_shapes(shape_specs, input_names):
    shapes = {}
    for spec in shape_specs or []:
        if not spec:
            continue
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        for part in parts:
            if ":" in part:
                name, shape_str = part.split(":", 1)
            elif "=" in part:
                name, shape_str = part.split("=", 1)
            else:
                name, shape_str = None, part
            dims = [int(x) for x in shape_str.strip().split("x") if x]
            if not dims:
                raise ValueError(f"Invalid shape spec: {part}")
            if name is None:
                if len(input_names) != 1:
                    raise ValueError(
                        f"Shape '{part}' does not name an input and there are {len(input_names)} inputs."
                    )
                name = input_names[0]
            shapes[name.strip()] = tuple(dims)
    return shapes


def _dtype_is_int(dtype):
    return dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)


def _make_input(shape, dtype, rng):
    if _dtype_is_int(dtype):
        return rng.integers(0, 1000, size=shape, dtype=dtype)
    return rng.random(size=shape, dtype=np.float32).astype(dtype)


def _binding_info(engine):
    info = []
    # TensorRT 10+ uses num_io_tensors instead of num_bindings
    num_bindings = engine.num_io_tensors if hasattr(engine, 'num_io_tensors') else engine.num_bindings
    for i in range(num_bindings):
        if hasattr(engine, 'get_tensor_name'):
            # TensorRT 10+ API
            name = engine.get_tensor_name(i)
            info.append(
                {
                    "index": i,
                    "name": name,
                    "is_input": engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
                    "dtype": str(engine.get_tensor_dtype(name)),
                    "shape": tuple(engine.get_tensor_shape(name)),
                    "is_shape_binding": False,  # Shape tensors handled differently in TRT 10+
                }
            )
        else:
            # Legacy API
            info.append(
                {
                    "index": i,
                    "name": engine.get_binding_name(i),
                    "is_input": engine.binding_is_input(i),
                    "dtype": str(engine.get_binding_dtype(i)),
                    "shape": tuple(engine.get_binding_shape(i)),
                    "is_shape_binding": engine.is_shape_binding(i),
                }
            )
    return info


def _alloc_binding(engine, binding_idx, shape):
    dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
    size = int(np.prod(shape))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    return host_mem, device_mem, dtype


def main():
    parser = argparse.ArgumentParser(description="Native TensorRT engine benchmark")
    parser.add_argument("--engine", required=True, help="Path to TensorRT engine (.plan)")
    parser.add_argument("--runs", type=int, default=100, help="Timed runs")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (implicit batch engines only)")
    parser.add_argument(
        "--shapes",
        action="append",
        default=[],
        help="Input shapes, e.g. input_ids:1x128,attention_mask:1x128",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Print binding info")
    args = parser.parse_args()

    engine_path = os.path.abspath(args.engine)
    if not os.path.exists(engine_path):
        raise SystemExit(f"Engine not found: {engine_path}")

    with open(engine_path, "rb") as f:
        engine_data = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    if engine is None:
        raise SystemExit("Failed to deserialize engine. Check TensorRT version compatibility.")

    if args.verbose:
        print("Bindings:")
        for info in _binding_info(engine):
            print(info)

    context = engine.create_execution_context()
    if context is None:
        raise SystemExit("Failed to create execution context")

    # Detect TensorRT API version
    use_new_api = hasattr(engine, 'num_io_tensors')
    
    if use_new_api:
        # TensorRT 10+ API
        num_bindings = engine.num_io_tensors
        input_names = [engine.get_tensor_name(i) for i in range(num_bindings) 
                      if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        input_indices = [i for i in range(num_bindings) 
                        if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
    else:
        # Legacy API
        num_bindings = engine.num_bindings
        input_indices = [i for i in range(num_bindings) if engine.binding_is_input(i)]
        input_names = [engine.get_binding_name(i) for i in input_indices]
    
    shape_overrides = _parse_shapes(args.shapes, input_names)

    # Check if engine has implicit batch dimension (legacy engines only)
    if use_new_api or not hasattr(engine, 'has_implicit_batch_dimension'):
        batch_size = None  # TensorRT 10+ doesn't use implicit batch
    elif engine.has_implicit_batch_dimension:
        batch_size = args.batch
    else:
        batch_size = None

    # Configure binding shapes for explicit batch engines
    if not use_new_api and hasattr(engine, 'has_implicit_batch_dimension') and not engine.has_implicit_batch_dimension:
        for i in input_indices:
            if engine.is_shape_binding(i):
                continue
            name = engine.get_binding_name(i)
            binding_shape = engine.get_binding_shape(i)
            if any(dim < 0 for dim in binding_shape):
                if name not in shape_overrides:
                    raise SystemExit(f"Dynamic input '{name}' needs --shapes")
                binding_shape = shape_overrides[name]
            context.set_binding_shape(i, binding_shape)
    elif use_new_api:
        # TensorRT 10+ API: set input shapes
        for name in input_names:
            tensor_shape = engine.get_tensor_shape(name)
            if any(dim < 0 for dim in tensor_shape):
                if name not in shape_overrides:
                    raise SystemExit(f"Dynamic input '{name}' needs --shapes")
                tensor_shape = shape_overrides[name]
            context.set_input_shape(name, tensor_shape)

    # Validate output shapes
    if use_new_api:
        for i in range(num_bindings):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                continue
            shape = context.get_tensor_shape(name)
            if any(dim < 0 for dim in shape):
                raise SystemExit(f"Output '{name}' has unresolved shape: {shape}")
    else:
        for i in range(num_bindings):
            if engine.binding_is_input(i):
                continue
            shape = context.get_binding_shape(i)
            if any(dim < 0 for dim in shape):
                raise SystemExit(f"Output '{engine.get_binding_name(i)}' has unresolved shape: {shape}")

    rng = np.random.default_rng(args.seed)
    stream = cuda.Stream()

    bindings_int = [None] * num_bindings  # Integer addresses for TensorRT
    bindings_ptr = [None] * num_bindings  # Device pointers for CUDA
    host_inputs = {}
    host_outputs = {}
    input_shapes = {}
    output_shapes = {}

    # Allocate inputs
    for i in input_indices:
        if use_new_api:
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            if any(dim < 0 for dim in shape):
                shape = shape_overrides[name]
            dtype = trt.nptype(engine.get_tensor_dtype(name))
        else:
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            if engine.has_implicit_batch_dimension:
                shape = (batch_size,) + tuple(shape)
            elif engine.is_shape_binding(i):
                if name not in shape_overrides:
                    raise SystemExit(f"Shape binding '{name}' needs --shapes")
                shape = shape_overrides[name]
            dtype = trt.nptype(engine.get_binding_dtype(i))
        
        size = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        host_inputs[name] = host_mem
        input_shapes[name] = tuple(shape)
        bindings_int[i] = int(device_mem)
        bindings_ptr[i] = device_mem

        data = _make_input(shape, dtype, rng)
        np.copyto(host_mem, data.ravel())

    # Allocate outputs
    for i in range(num_bindings):
        if use_new_api:
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                continue
            shape = context.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
        else:
            if engine.binding_is_input(i):
                continue
            name = engine.get_binding_name(i)
            shape = context.get_binding_shape(i)
            if engine.has_implicit_batch_dimension:
                shape = (batch_size,) + tuple(shape)
            dtype = trt.nptype(engine.get_binding_dtype(i))
        
        size = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        host_outputs[name] = host_mem
        output_shapes[name] = tuple(shape)
        bindings_int[i] = int(device_mem)
        bindings_ptr[i] = device_mem

    def _run_once():
        if use_new_api:
            # TensorRT 10+ API: set tensor addresses
            for name in input_names:
                idx = [i for i in range(num_bindings) if engine.get_tensor_name(i) == name][0]
                context.set_tensor_address(name, bindings_int[idx])
            for i in range(num_bindings):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    context.set_tensor_address(name, bindings_int[i])
            
            # Copy inputs to device
            for name in input_names:
                idx = [i for i in range(num_bindings) if engine.get_tensor_name(i) == name][0]
                cuda.memcpy_htod_async(bindings_ptr[idx], host_inputs[name], stream)
            
            # Execute
            context.execute_async_v3(stream_handle=stream.handle)
            
            # Copy outputs to host
            for i in range(num_bindings):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                    cuda.memcpy_dtoh_async(host_outputs[name], bindings_ptr[i], stream)
        else:
            # Legacy API
            for i in input_indices:
                cuda.memcpy_htod_async(bindings_ptr[i], host_inputs[engine.get_binding_name(i)], stream)
            context.execute_async_v2(bindings=bindings_int, stream_handle=stream.handle)
            for i in range(num_bindings):
                if engine.binding_is_input(i):
                    continue
                cuda.memcpy_dtoh_async(host_outputs[engine.get_binding_name(i)], bindings_ptr[i], stream)
        stream.synchronize()

    # Warmup
    for _ in range(args.warmup):
        _run_once()

    latencies = []
    for _ in range(args.runs):
        start = time.perf_counter()
        _run_once()
        end = time.perf_counter()
        latencies.append(end - start)

    lat = np.array(latencies)
    mean_ms = float(lat.mean() * 1000)
    std_ms = float(lat.std() * 1000)
    p50_ms = float(np.percentile(lat, 50) * 1000)
    p90_ms = float(np.percentile(lat, 90) * 1000)
    p99_ms = float(np.percentile(lat, 99) * 1000)

    total_infers = args.runs * (batch_size if batch_size is not None else 1)
    throughput = float(total_infers / lat.sum())

    summary = {
        "host": socket.gethostname(),
        "engine": os.path.basename(engine_path),
        "engine_path": engine_path,
        "runs": args.runs,
        "warmup": args.warmup,
        "batch": batch_size,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "mean_latency_ms": mean_ms,
        "std_latency_ms": std_ms,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "p99_ms": p99_ms,
        "throughput": throughput,
    }

    print(f"Runs: {summary['runs']}")
    print(f"Mean latency: {mean_ms:.2f} ms")
    print(f"Std latency: {std_ms:.2f} ms")
    print(f"P50: {p50_ms:.2f} ms")
    print(f"P90: {p90_ms:.2f} ms")
    print(f"P99: {p99_ms:.2f} ms")
    print(f"Throughput: {throughput:.2f} inferences/sec")

    out_path = args.out
    if out_path is None:
        out_dir = os.path.join("results", "trt-native", "benchmarks")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{os.path.basename(engine_path)}_bench.json")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
