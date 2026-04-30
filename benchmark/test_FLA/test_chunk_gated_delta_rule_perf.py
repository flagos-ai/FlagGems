"""
Performance benchmark for chunk_gated_delta_rule_fwd operator.

This benchmark tests the chunk_gated_delta_rule_fwd operator which implements
the chunk-based gated delta rule used in models like Qwen3-Next.

Per FlagGems Operator Development Competition Requirements:
- Section 3.3.2: Performance Competitiveness (20%)
- Section 3.1.4: Test case coverage requirements (input sizes, dimensions, parameters)

Reference:
- Qwen3-Next: https://huggingface.co/Qwen
- GatedDeltaNet: https://github.com/NVlabs/GatedDeltaNet
"""
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

import flag_gems
from benchmark.base import Benchmark
from benchmark.consts import FLOAT_DTYPES

# Import the operator
from flag_gems.fused import chunk_gated_delta_rule

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(project_root)
sys.path.append(project_root)

# Manually create GatedDeltaNet module hierarchy to work around circular import
gated_delta_net = types.ModuleType("GatedDeltaNet")
gated_delta_net.__path__ = [os.path.join(project_root, "GatedDeltaNet")]
sys.modules["GatedDeltaNet"] = gated_delta_net

lit_gpt = types.ModuleType("GatedDeltaNet.lit_gpt")
lit_gpt.__path__ = [os.path.join(project_root, "GatedDeltaNet", "lit_gpt")]
sys.modules["GatedDeltaNet.lit_gpt"] = lit_gpt

# Try to import reference implementations for comparison
GATEDDELTANET_AVAILABLE = False
FLA_AVAILABLE = False

try:
    from GatedDeltaNet.lit_gpt.gated_delta_rule_ops.chunk import (
        chunk_gated_delta_rule as reference_gateddnet,
    )

    GATEDDELTANET_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    print("GatedDeltaNet not available, skipping reference comparison")
    pass

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as reference_fla

    FLA_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    print("FLA not available, skipping reference comparison")
    pass


class ChunkGatedDeltaRuleBenchmark(Benchmark):
    """Benchmark for chunk_gated_delta_rule_fwd operator."""

    DEFAULT_DTYPES = [torch.float16, torch.bfloat16]

    def __init__(self, backend="gateddnet", with_initial_state=False, *args, **kwargs):
        """
        Initialize benchmark.

        Args:
            backend: 'fla' or 'gateddnet' - which backend implementation to test
            with_initial_state: Whether to test with initial_state parameter
        """
        super().__init__(*args, **kwargs)
        self.backend = backend
        self.with_initial_state = with_initial_state
        # Override shapes with custom shapes for this operator
        self.shapes = self.set_more_shapes()

    def set_shapes(self, shape_file_path=None):
        """
        Override set_shapes to prevent loading from shape file.

        Chunk gated delta rule requires specific (B, T, H, K, V) shapes.
        """
        # Simply use shapes already set in __init__
        pass

    def set_more_shapes(self):
        """
        Define test shapes covering small, medium, large sizes.

        Per competition requirements (intro,txt section 3.1.4):
        - Small sizes: 1x1, 8x8 scale
        - Medium sizes: 64x64, 256x256 scale
        - Large sizes: 1024x1024, 4096x4096 scale

        Format: (B, T, H, K, V)
        - B: Batch size
        - T: Sequence length
        - H: Number of heads
        - K: Key/Query head dimension
        - V: Value head dimension

        Note: Reduced some sizes to avoid GPU OOM errors during benchmarking.
        """
        # Small sizes (per competition requirement)
        small_shapes = [
            (2, 128, 4, 64, 32),  # Small: B=2, T=128, H=4, K=64, V=32
        ]

        # Medium sizes - reduced to avoid memory issues
        medium_shapes = [
            (2, 256, 4, 64, 32),  # Medium: B=2, T=256, H=4, K=64, V=32
            (2, 512, 4, 64, 32),  # Large: B=2, T=512, H=4, K=64, V=32
        ]

        # Large sizes - reduced H to avoid memory issues
        large_shapes = [
            (1, 1024, 2, 64, 32),  # Competition Large: T=1024, reduced H
            (1, 2048, 2, 64, 32),  # Competition XLarge: T=2048, reduced H
        ]

        return small_shapes + medium_shapes + large_shapes

    def set_more_metrics(self):
        """Add bandwidth metric for the operator."""
        return ["gbps"]

    def get_gbps(self, args, latency):
        """
        Calculate effective bandwidth in GB/s.

        Total I/O includes:
        - Input tensors: q, k, v, g, beta
        - Output tensor: o (same shape as v)

        Args:
            args: Tuple of (q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens)
                  Note: backend and BT are passed via kwargs, not args
            latency: Operation latency in milliseconds

        Returns:
            Bandwidth in GB/s
        """
        # Unpack positional arguments only (backend and BT are in kwargs, not used here)
        q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens = args

        # Calculate total input bytes
        input_bytes = (
            q.numel() * q.element_size()
            + k.numel() * k.element_size()
            + v.numel() * v.element_size()
            + g.numel() * g.element_size()
            + beta.numel() * beta.element_size()
        )

        # Output is o tensor, same shape as v
        output_bytes = v.numel() * v.element_size()

        io_amount = input_bytes + output_bytes
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, cur_dtype):
        """
        Generate input tensors with various parameters.

        Creates tensors in the correct layout for each backend:
        - 'gateddnet': head-first [B, H, T, K/V]
        - 'fla': time-first [B, T, H, K/V]

        Args:
            cur_dtype: Current dtype being tested

        Yields:
            Tuple of (q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, backend, BT)
        """
        for B, T, H, K, V in self.shapes:
            # Scale based on K dimension (standard attention scaling)
            scale = K**-0.5

            if self.backend == "gateddnet":
                # Head-first format [B, H, T, K/V] for GagedDeltaNet backend
                q = torch.randn(B, H, T, K, dtype=cur_dtype, device=self.device)
                k = F.normalize(
                    torch.randn(B, H, T, K, dtype=cur_dtype, device=self.device),
                    p=2,
                    dim=-1,
                )
                v = torch.randn(B, H, T, V, dtype=cur_dtype, device=self.device)
                g = F.logsigmoid(
                    torch.randn(B, H, T, dtype=cur_dtype, device=self.device)
                )
                beta = torch.rand(
                    B, H, T, dtype=cur_dtype, device=self.device
                ).sigmoid()
            else:  # 'fla'
                # Time-first format [B, T, H, K/V] for FLA backend
                q = torch.randn(B, T, H, K, dtype=cur_dtype, device=self.device)
                k = F.normalize(
                    torch.randn(B, T, H, K, dtype=cur_dtype, device=self.device),
                    p=2,
                    dim=-1,
                )
                v = torch.randn(B, T, H, V, dtype=cur_dtype, device=self.device)
                g = F.logsigmoid(
                    torch.randn(B, T, H, dtype=cur_dtype, device=self.device)
                )
                beta = torch.rand(
                    B, T, H, dtype=cur_dtype, device=self.device
                ).sigmoid()

            # Initial state (if testing with it)
            initial_state = None
            if self.with_initial_state:
                # initial_state is always [B, H, K, V] format
                initial_state = torch.randn(
                    B, H, K, V, dtype=cur_dtype, device=self.device
                )

            # Test different BT (chunk size) values
            # Per competition requirements: test parameter variations
            for BT in [16, 32, 64, 128, 256]:
                # Ensure T is divisible by BT
                if T % BT != 0:
                    continue

                # Yield tensors and positional args, then kwargs as dict
                # Note: strings are not unpacked to args by unpack_to_args_kwargs,
                # so we pass backend/BT via kwargs dict (like ctc_loss benchmark does)
                yield (
                    q,
                    k,
                    v,
                    g,
                    beta,
                    scale,  # scale parameter (K^-0.5)
                    initial_state,
                    False,  # output_final_state
                    None,  # cu_seqlens (not supported yet)
                    {
                        "backend": self.backend,
                        "BT": BT,
                    },
                )

    def run(self):
        """
        Override run() to handle unavailable reference implementations.

        When reference is unavailable:
        - Set latency_base to None (displays "N/A" in table)
        - Skip speedup calculation
        - Continue testing FlagGems implementation normally
        """
        import gc

        from .conftest import Config, emit_record_logger
        from .consts import BenchmarkMetrics, BenchmarkResult, OperationAttribute

        if Config.query:
            self.init_default_config()
            attri = OperationAttribute(
                op_name=self.op_name,
                recommended_core_shapes=self.shapes,
                shape_desc=self.shape_desc,
            )
            print(attri)
            emit_record_logger(attri.to_dict())
            return

        self.init_user_config()

        for dtype in self.to_bench_dtypes:
            metrics = []
            input_iter = self.get_input_iter(dtype)

            done = False
            while not done:
                try:
                    input = next(input_iter)
                except StopIteration:
                    done = True
                    continue
                except (RuntimeError, Exception) as e:
                    print(
                        f"\033[31mFAILED\033[0m: Operator={self.op_name} "
                        f"dtype={dtype} err=<<<{e}>>>"
                    )
                    pytest.fail(str(e))

                metric = BenchmarkMetrics()
                try:
                    args, kwargs = self.unpack_to_args_kwargs(input)
                    metric.shape_detail = self.record_shapes(*args, **kwargs)

                    # Check if reference is available before measuring latency
                    if "latency_base" in self.to_bench_metrics:
                        # Try to call torch_op once to check availability
                        try:
                            ref_result = self.torch_op(*args, **kwargs)
                            if ref_result is None:
                                # Reference unavailable or failed
                                metric.latency_base = None
                            else:
                                # Reference available, measure latency
                                metric.latency_base = self.get_latency(
                                    self.torch_op, *args, **kwargs
                                )
                        except (RuntimeError, Exception):
                            # Reference failed during execution
                            metric.latency_base = None

                    if "latency" in self.to_bench_metrics:
                        if self.gems_op:
                            metric.latency = self.get_latency(
                                self.gems_op, *args, **kwargs
                            )
                        else:
                            with flag_gems.use_gems(exclude=["zero_"]):
                                metric.latency = self.get_latency(
                                    self.torch_op, *args, **kwargs
                                )

                    # Skip speedup calculation if latency_base is None
                    if "speedup" in self.to_bench_metrics:
                        if metric.latency_base is not None:
                            metric.speedup = metric.latency_base / metric.latency
                        else:
                            metric.speedup = None

                    if "gbps" in self.to_bench_metrics:
                        metric.gbps_base = self.get_gbps(
                            args, latency=metric.latency_base
                        )
                        metric.gbps = self.get_gbps(args, latency=metric.latency)

                    if "tflops" in self.to_bench_metrics:
                        metric.tflops = (
                            self.get_tflops(self.torch_op, *args, **kwargs)
                            / metric.latency
                            / 1e12
                            * 1e3
                        )

                except (RuntimeError, Exception) as e:
                    metric.error_msg = str(e)
                    pytest.fail(str(e))
                finally:
                    metrics.append(metric)
                    gc.collect()

            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics,
            )
            print(result)
            emit_record_logger(result.to_json())


def _torch_op_wrapper(*args, **kwargs):
    """
    Wrapper for torch/reference implementation.

    Uses available reference implementation (GatedDeltaNet or FLA) if available.
    Returns None when reference is unavailable or fails, which signals the benchmark
    to skip latency measurement for the reference implementation.

    This allows the benchmark to:
    1. Compare FlagGems against reference implementations when available
    2. Skip reference measurement when unavailable (displays "N/A" in table)
    3. Continue benchmarking FlagGems even when reference fails
    """
    # Unpack positional arguments (tensors and scalars)
    # Format: (q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens)
    q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens = args

    # Unpack keyword arguments (backend and BT passed via dict)
    backend = kwargs.get("backend", "gateddnet")
    BT = kwargs.get("BT", 64)

    # Try to use reference implementation if available
    if backend == "fla" and FLA_AVAILABLE:
        # FLA uses different parameter order: (q, k, v, g, beta, scale, ...)
        # Note: FLA expects time-first layout [B, T, H, K/V]
        try:
            o, final_state = reference_fla(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )
            return None, o, None, final_state, None, None, None
        except RuntimeError as e:
            # Check for CUDA OOM specifically
            if "out of memory" in str(e).lower() or "out of resource" in str(e).lower():
                return None  # Signal: reference failed (OOM)
            # Other runtime errors - let them propagate
            raise
        except Exception:
            # Unexpected errors - let them propagate
            raise

    elif backend == "gateddnet" and GATEDDELTANET_AVAILABLE:
        # GatedDeltaNet uses: (q, k, v, beta, g, BT, ...)
        # Note: GatedDeltaNet expects head-first layout [B, H, T, K/V]
        try:
            o, final_state = reference_gateddnet(
                q,
                k,
                v,
                beta,
                g,
                BT=BT,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )
            return g, o, None, final_state, None, None, None
        except RuntimeError as e:
            # Check for CUDA OOM specifically
            if "out of memory" in str(e).lower() or "out of resource" in str(e).lower():
                return None  # Signal: reference failed (OOM)
            # Other runtime errors - let them propagate
            raise
        except Exception:
            # Unexpected errors - let them propagate
            raise

    else:
        # Reference implementation not available at all
        # Return None to signal benchmark to skip latency measurement
        return None


# ============================================================================
# Pytest Test Functions
# ============================================================================


@pytest.mark.skipif(flag_gems.device != "cuda", reason="benchmark requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("backend", ["fla", "gateddnet"])
def test_perf_chunk_gated_delta_rule_basic(dtype, backend):
    """
    Benchmark chunk_gated_delta_rule_fwd with basic configurations.

    Tests all data types (float16, float32, bfloat16) and both backends
    without initial state parameter.
    """
    bench = ChunkGatedDeltaRuleBenchmark(
        backend=backend,
        with_initial_state=False,
        op_name="chunk_gated_delta_rule",
        torch_op=_torch_op_wrapper,
        dtypes=[dtype],
    )
    bench.set_gems(chunk_gated_delta_rule)
    bench.run()


@pytest.mark.skipif(flag_gems.device != "cuda", reason="benchmark requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("backend", ["fla", "gateddnet"])
def test_perf_chunk_gated_delta_rule_with_state(dtype, backend):
    """
    Benchmark chunk_gated_delta_rule_fwd with initial state.

    Tests all data types and both backends with initial_state parameter.
    This tests the stateful variant of the operator.
    """
    bench = ChunkGatedDeltaRuleBenchmark(
        backend=backend,
        with_initial_state=True,
        op_name="chunk_gated_delta_rule",
        torch_op=_torch_op_wrapper,
        dtypes=[dtype],
    )
    bench.set_gems(chunk_gated_delta_rule)
    bench.run()


@pytest.mark.skipif(flag_gems.device != "cuda", reason="benchmark requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("backend", ["fla", "gateddnet"])
def test_perf_chunk_gated_delta_rule_bfloat16_only(backend):
    """
    Benchmark with bfloat16 only (typical for LLMs).

    This is a focused benchmark for the most commonly used dtype in
    large language model training.
    """
    bench = ChunkGatedDeltaRuleBenchmark(
        backend=backend,
        with_initial_state=False,
        op_name="chunk_gated_delta_rule",
        torch_op=_torch_op_wrapper,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(chunk_gated_delta_rule)
    bench.run()


@pytest.mark.skipif(flag_gems.device != "cuda", reason="benchmark requires CUDA device")
@pytest.mark.chunk_gated_delta_rule
@pytest.mark.parametrize("backend", ["gateddnet"])
def test_perf_chunk_gated_delta_rule_gateddnet_comprehensive(backend):
    """
    Comprehensive benchmark for GagedDeltaNet backend.

    Tests all dtypes with the GagedDeltaNet backend, which is the
    optimized FlagGems implementation.
    """
    bench = ChunkGatedDeltaRuleBenchmark(
        backend=backend,
        with_initial_state=False,
        op_name="chunk_gated_delta_rule",
        torch_op=_torch_op_wrapper,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(chunk_gated_delta_rule)
    bench.run()


if __name__ == "__main__":
    """
    Direct execution entry point.

    Run benchmarks for all backends and data types, displaying results to console.
    """
    print(f"Using device: {flag_gems.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    print("\n" + "=" * 80)
    print("CHUNK_GATED_DELTA_RULE PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Run benchmarks for both backends
    for backend in ["gateddnet", "fla"]:
        print(f"\n{'=' * 80}")
        print(f"Backend: {backend}")
        print("=" * 80)

        for dtype in FLOAT_DTYPES:
            print(f"\nTesting dtype: {dtype}")
            try:
                test_perf_chunk_gated_delta_rule_basic(dtype, backend)
            except Exception as e:
                print(f"Error testing {dtype} with {backend}: {e}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
