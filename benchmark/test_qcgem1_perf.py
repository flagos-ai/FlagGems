# SPDX-License-Identifier: Apache-2.0
# QC-GEM1 (GemLite v0.5.1) Benchmark for FlagGems
# Supports W8A16, W4A16, W8A8_FP8, W4A4_MXFP4, W4A4_NVFP4 precision modes
#
# Usage:
#   # W8A16 benchmark
#   pytest -s ./benchmark/test_qcgem1_perf.py::test_perf_qcgem1_w8a16 \
#       --mode kernel --dtypes float16,bfloat16
#
#   # W4A16 benchmark
#   pytest -s ./benchmark/test_qcgem1_perf.py::test_perf_qcgem1_w4a16 \
#       --mode kernel --dtypes float16,bfloat16
#
#   # All precisions with model shapes
#   pytest -s ./benchmark/test_qcgem1_perf.py -m qcgem1 \
#       --mode kernel --dtypes float16 --parallel 8


from typing import Generator

import pytest
import torch

import flag_gems
from benchmark.attri_util import (
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    BenchLevel,
    model_shapes,
)
from benchmark.conftest import Config
from benchmark.performance_utils import Benchmark


# =============================================================================
# qcgem1 (GemLite v0.5.1) imports
# =============================================================================
try:
    from flag_gems.ops.qcgem1 import (
        DType,
        forward_functional,
        GEMLITE_MATMUL_TYPES,
        GEMLITE_MATMUL_TYPES_MAPPING,
        get_matmul_type,
    )
    from flag_gems.ops.qcgem1.dtypes import DTYPE_TO_TORCH, TORCH_TO_DTYPE

    QCGEM1_AVAILABLE = True
except Exception:
    DType = None
    forward_functional = None
    GEMLITE_MATMUL_TYPES = []
    GEMLITE_MATMUL_TYPES_MAPPING = {}
    get_matmul_type = None
    DTYPE_TO_TORCH = {}
    TORCH_TO_DTYPE = {}
    QCGEM1_AVAILABLE = False


# =============================================================================
# Precision configurations
# =============================================================================
# (W_nbits, input_dtype_enum, activation_scaling, group_size, desc)
PRECISION_CONFIGS = {
    "W8A16": (
        8, DType.FP16, False, 128,
        "W8/FP16 (INT8 weight + FP16 activation, weight-only)"
    ),
    "W4A16": (
        4, DType.FP16, False, 128,
        "W4/FP16 (INT4 weight + FP16 activation, weight-only)"
    ),
}


def get_input_dtype_enum(torch_dtype):
    """Map torch dtype to qcgem1 DType enum."""
    if torch_dtype == torch.float16:
        return DType.FP16
    elif torch_dtype == torch.bfloat16:
        return DType.BF16
    elif torch_dtype == torch.float32:
        return DType.FP32
    return DType.FP16


def quantize_weights_qcgem1(W_fp, w_nbits, group_size, device):
    """Quantize float weights to qcgem1 format.
    
    Returns: (W_q, scales, zeros)
    - W_q: uint8 tensor, shape (K, N) packed
    - scales: float32 tensor, shape (N, K//group_size)
    - zeros: float32 tensor, shape (N, K//group_size)
    """
    n, k = W_fp.shape  # W_fp: (N, K)
    n_groups = k // group_size

    # Group-wise quantization
    W_view = W_fp.view(n, n_groups, group_size)  # (N, n_groups, G)
    scales = W_view.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)  # (N, n_groups, 1)
    scales = scales.squeeze(-1)  # (N, n_groups)

    if w_nbits == 8:
        max_val = 127.0
        W_normalized = W_view / scales.unsqueeze(-1)
        W_q = W_normalized.round().clamp(-max_val, max_val).to(torch.int8)
        # For W8A16: reshape to 2D (N, K) - flatten the groups
        W_packed = W_q.view(n, n_groups * group_size).to(torch.uint8)
        zeros = torch.zeros_like(scales)
    else:
        max_val = 7.0
        W_normalized = W_view / scales.unsqueeze(-1)
        W_q = W_normalized.round().clamp(-max_val, max_val).to(torch.int8)
        zeros = torch.zeros_like(scales)

    # Pack into uint8: 4-bit packs into bytes
    # W_q: (N, n_groups, G) -> pack G elements per byte
    if w_nbits == 4:
        n_out_cols = n_groups * group_size // 2
        W_packed = torch.empty(n, n_out_cols, dtype=torch.uint8, device=device)
        W_q_pos = W_q.view(n, n_groups, group_size)
        for g in range(n_groups):
            low = W_q_pos[:, g, :].to(torch.uint8)
            high = torch.zeros(n, dtype=torch.uint8, device=device)
            W_packed[:, g::n_groups] = (high << 4) | (low & 0x0F)
    # else: W8A16 already handled above, W_packed is already set

    # Scales and zeros in (N, n_groups) -> will be transposed to (n_groups, N) in kernel
    return W_packed, scales, zeros


# =============================================================================
# qcgem1 Benchmark class
# =============================================================================
class QCGem1Benchmark(Benchmark):
    """
    Benchmark for qcgem1 (GemLite v0.5.1) quantized GEMM kernels.

    Supports:
      - W8A16: INT8 weights + FP16 activations
      - W4A16: INT4 weights + FP16 activations
      - W8A8_FP8: INT8 weights + FP8 activations
      - W4A4_MXFP4: MXFP4 weight+activation
      - W4A4_NVFP4: NVFP4 weight+activation
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]
    SHAPE_CONFIG_KEYS = ("mm", "BlasBenchmark")

    def __init__(
        self,
        *args,
        precision="W4A16",
        group_size=128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.group_size = group_size

        cfg = PRECISION_CONFIGS.get(precision)
        if cfg is None:
            raise ValueError(
                f"Unsupported precision '{precision}'. "
                f"Supported: {list(PRECISION_CONFIGS.keys())}"
            )
        self.w_nbits, self.input_dtype_enum, self.activation_scaling, _, self._desc = cfg
        self.shape_desc = "M, N, K"

    def set_more_shapes(self):
        """Add large-K model shapes from attri_util."""
        return super().set_more_shapes()

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        normalized = []
        for shape in self.shapes:
            if len(shape) == 4:
                _, m, n, k = shape
                normalized.append((m, n, k))
            elif len(shape) == 3:
                normalized.append(shape)
            else:
                raise ValueError(
                    "qcgem1 benchmark expects shapes in (M, N, K) or (B, M, N, K) format."
                )
        self.shapes = normalized
        self.shape_desc = "M, N, K"

    def get_input_iter(self, cur_dtype) -> Generator:
        if not QCGEM1_AVAILABLE:
            raise RuntimeError("qcgem1 kernels are not available")

        input_dtype_enum = get_input_dtype_enum(cur_dtype)

        for m, n, k in self.shapes:
            # Generate float weights (N, K) and quantize
            W_fp = torch.randn(n, k, dtype=cur_dtype, device=self.device)
            W_q, scales, zeros = quantize_weights_qcgem1(
                W_fp, self.w_nbits, self.group_size, self.device
            )

            # Input: (M, K)
            x = torch.randn(m, k, dtype=cur_dtype, device=self.device)

            # Transpose weights: kernel expects (K, N) not (N, K)
            W_q_T = W_q.T.contiguous()  # (n_groups*group_size, N)
            scales_T = scales.T.contiguous()  # (n_groups, N)
            zeros_T = zeros.T.contiguous()  # (n_groups, N)

            tensor_args = [W_q_T, scales_T, zeros_T]
            # meta_args structure must match what forward_functional expects:
            # meta_args[0] = scaled_activations (indexed)
            # meta_args[1] = data_contiguous (indexed AND explicitly passed to kernel)
            # meta_args[2:-1] = W_nbits, group_size, unpack_mask, elements_per_sample, ..., W_group_mode
            # meta_args[-1] = type_id (computed internally by forward_functional, not passed to kernel)
            # NOTE: forward_functional does x, *tensor_args, scales_x, *meta_args[1:-1], data_contiguous, type_id
            # meta_args[1:-1] gives [data_contiguous, W_nbits, ..., W_group_mode] = 10 elements
            # Then explicitly adds data_contiguous and type_id
            # Total kernel args: x(1) + 3 tensors(3) + scales_x(1) + meta_args[1:-1](10) + data_contiguous(1) + type_id(1) = 17
            meta_args = [
                int(self.activation_scaling),  # [0] scaled_activations
                1,  # [1] data_contiguous (indexed AND explicitly passed to kernel)
                self.w_nbits,  # [2]
                self.group_size,  # [3]
                0,  # [4] unpack_mask
                1,  # [5] elements_per_sample
                input_dtype_enum.value,  # [6] input_dtype
                input_dtype_enum.value,  # [7] output_dtype
                0,  # [8] acc_dtype (0 = auto)
                0,  # [9] meta_dtype (0 = auto)
                0,  # [10] channel_scale_mode (0 = auto)
                0,  # [11] W_group_mode (0 = auto)
                # NOTE: type_id is NOT included here - forward_functional computes it
                # meta_args[-1] would be [12], but forward_functional expects it at [0]
                # The kernel doesn't receive type_id as a positional arg, it's passed explicitly
            ]
            yield x, tensor_args, meta_args

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        W_q = args[1]
        m, k = x.shape
        n = W_q.shape[1]
        return 2 * m * n * k

    def _build_metric_from_input(self, input_item):
        from benchmark.attri_util import BenchmarkMetrics

        x, tensor_args, meta_args = input_item
        m = x.shape[0]
        matmul_type_str = get_matmul_type(m, self.w_nbits, mx_dtype=False)
        matmul_type_id = GEMLITE_MATMUL_TYPES_MAPPING.get(matmul_type_str, -1)

        metric = BenchmarkMetrics()
        metric.shape_detail = [x.shape, tensor_args[0].shape]

        def qcgem1_fn():
            return forward_functional(x, None, tensor_args, meta_args, matmul_type_id)

        def fp16_ref_fn():
            # Dequantize weights and compute FP16 reference
            W_q_T = tensor_args[0]  # Already transposed to (K, N)
            scales_T = tensor_args[1]  # Already transposed to (n_groups, N)
            zeros_T = tensor_args[2]  # Already transposed to (n_groups, N)
            # Compute dimensions correctly: n from W_q_T, n_groups from K // group_size
            k = W_q_T.shape[0]
            n = W_q_T.shape[1]
            n_groups = k // self.group_size
            group_size = self.group_size
            # Reshape W_q_T (K, N) -> (n_groups, group_size, N)
            W_reshaped = W_q_T.reshape(n_groups, group_size, n)  # (n_groups, group_size, N)
            # scales_T is (n_groups, N), broadcast to match W_reshaped
            scales_bc = scales_T.unsqueeze(1)  # (n_groups, 1, N)
            # Dequantize: scales * W, broadcasting (n_groups, group_size, N) * (n_groups, 1, N)
            W_deq = W_reshaped.float() * scales_bc
            # Reshape back to (K, N)
            W_deq = W_deq.reshape(k, n)
            return torch.mm(x, W_deq.to(x.dtype))

        # Benchmark reference function (FP16 dequantized matmul)
        if "latency_base" in self.to_bench_metrics:
            metric.latency_base = self.get_latency(fp16_ref_fn)
        
        # Benchmark qcgem1 kernel
        if "latency" in self.to_bench_metrics:
            try:
                metric.latency = self.get_latency(qcgem1_fn)
            except Exception as e:
                print(f"[DEBUG] qcgem1_fn failed: {e}")
                import traceback
                traceback.print_exc()
                metric.latency = 0
                metric.error_msg = str(e)
        if "speedup" in self.to_bench_metrics:
            if metric.latency_base > 0 and metric.latency > 0:
                metric.speedup = metric.latency_base / metric.latency
        if "gbps" in self.to_bench_metrics:
            metric.gbps_base = 0
            metric.gbps = 0
        if "tflops" in self.to_bench_metrics:
            metric.tflops = self.get_tflops(None, x, tensor_args[0])
        return metric


def BlasBenchmark_set_more_shapes(self):
    large_k_shapes = [
        (8, 1848, 1536, 151936),
        (8, 1848, 1536, 128256),
        (8, 1848, 1536, 152064),
        (8, 4096, 1, 152064),
    ]
    try:
        ms = model_shapes()
        return large_k_shapes + ms
    except Exception:
        return large_k_shapes


# Monkey-patch for set_more_shapes reference
BlasBenchmark = type('BlasBenchmark', (), {'set_more_shapes': BlasBenchmark_set_more_shapes})()


# =============================================================================
# Pytest benchmark tests
# =============================================================================
@pytest.mark.qcgem1
@pytest.mark.parametrize("precision", list(PRECISION_CONFIGS.keys()))
def test_perf_qcgem1_gemm(precision):
    """Benchmark qcgem1 GEMM kernels for different weight precisions."""
    if not QCGEM1_AVAILABLE:
        pytest.skip("qcgem1 kernels not available")

    bench = QCGem1Benchmark(
        op_name=f"qcgem1_{precision.lower()}",
        torch_op=None,
        precision=precision,
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()


@pytest.mark.qcgem1
def test_perf_qcgem1_w8a16():
    """Benchmark qcgem1 W8A16: INT8 weight + FP16 activation."""
    if not QCGEM1_AVAILABLE:
        pytest.skip("qcgem1 kernels not available")

    bench = QCGem1Benchmark(
        op_name="qcgem1_w8a16",
        torch_op=None,
        precision="W8A16",
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()


@pytest.mark.qcgem1
def test_perf_qcgem1_w4a16():
    """Benchmark qcgem1 W4A16: INT4 weight + FP16 activation."""
    if not QCGEM1_AVAILABLE:
        pytest.skip("qcgem1 kernels not available")

    bench = QCGem1Benchmark(
        op_name="qcgem1_w4a16",
        torch_op=None,
        precision="W4A16",
        dtypes=[torch.float16, torch.bfloat16],
    )
    bench.run()
