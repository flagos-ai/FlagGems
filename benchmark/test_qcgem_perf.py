# SPDX-License-Identifier: Apache-2.0
# QC-GEM: Benchmark for quantized matrix multiplication in FlagGems
# Supports w8A16, w4A16, w8A8 quantization configurations
# Usage:
#   pytest -s ./benchmark/test_qcgem_perf.py::test_qcgem_w8a16_benchmark -m qcgem_w8a16 \
#       --shape_file models_shapes/Qwen3.5-397B-A17B-p32768d1024.yaml --mode kernel \
#       --dtypes float16,bfloat16 --parallel 8


from typing import Generator, List, Tuple

import pytest
import torch

import flag_gems
from benchmark.attri_util import (
    DEFAULT_METRICS,
    FLOAT_DTYPES,
)
from benchmark.conftest import Config
from benchmark.performance_utils import Benchmark
from benchmark.weight_quantize import get_weight_manager, WeightQuantizer


# Quantization configurations mapping
QUANT_CONFIGS = {
    "w8A16": {"w_nbits": 8, "group_size": 128, "desc": "8-bit weight, 16-bit activation(INT8-FP16)"},   # 适用于原生FP16模型
    "w4A16": {"w_nbits": 4, "group_size": 128, "desc": "4-bit weight, 16-bit activation(INT4-FP16)"},   # 适用于原生FP16模型
    # "w4A8": {"w_nbits": 4, "group_size": 128, "desc": "4-bit weight, 8-bit activation(INT4-INT8)"},   # 适用于原生FP16模型
    # "w8A8": {"w_nbits": 8, "group_size": 128, "desc": "8-bit weight, 8-bit activation (INT8-INT8)"},    # 适用于原生FP16模型
    # "w4A4": {"w_nbits": 4, "group_size": 128, "desc": "4-bit weight, 4-bit activation(INT4-INT4)"},

    # "w8A8": {"w_nbits": 8, "group_size": 128, "desc": "8-bit weight, 8-bit activation (FP8-FP8)"},      # 适用于原生FP8模型
    # # "w4A8": {"w_nbits": 4, "group_size": 128, "desc": "4-bit weight, 8-bit activation (FP4-FP8)"},    # 适用于原生FP8模型
    # # "w4A4": {"w_nbits": 4, "group_size": 128, "desc": "4-bit weight, 4-bit activation(FP4-FP4)"},     # 适用于原生FP8模型
}


class QCGEMBenchmark(Benchmark):
    """
    Benchmark for QC-GEM quantized matrix multiplication.

    This compares:
    1. QC-GEM (quantized GEMM using INT4/INT8 weights)
    2. Reference GEMM (dequantized weights in FP16/BF16)

    Shape semantics (from YAML shape_desc: "B, M, N, K"):
    - Input x: (M, K) = (seq_len * batch, hidden_dim)
    - Weight W: (N, K) = (intermediate_dim, hidden_dim)
    - Output: (M, N)

    For GEMM: y = x @ W.t() where x shape is (M, K), W shape is (N, K)
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops", "speedup", "gbps"]

    def __init__(
        self,
        *args,
        input_fn,
        w_nbits: int = 4,
        group_size: int = 128,
        quant_config: str = "w4A16",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn
        self.w_nbits = w_nbits
        self.group_size = group_size
        self.quant_config = quant_config
        # Initialize weight manager for caching
        self.weight_manager = get_weight_manager()

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            # Handle both (m, n, k) and (b, m, n, k) shape formats
            if len(shape) == 3:
                m, n, k = shape
                b = 1  # Default batch size
            else:
                b, m, n, k = shape
            yield from self.input_fn(
                b, m, n, k, cur_dtype, self.device,
                self.w_nbits, self.group_size, self.quant_config
            )

    def set_more_shapes(self):
        return []  # Only use shapes from shape file

    def get_tflops(self, op, *args, **kwargs):
        # FLOPs = 2 * M * K * N (matrix multiplication: M*K*N multiplications + M*K*N additions)
        if self.op_name.startswith("qcgem"):
            # args for qcgem: (inp, W_q, scales, zeros, ...)
            # x shape: (M, K), W_q shape: (N, K)
            x = args[0]
            W_q = args[1]
            m = x.shape[0]
            k = x.shape[1]
            n = W_q.shape[0]
            total_flops = m * k * n * 2
        else:
            total_flops = 0
        return total_flops

    def get_gbps(self, args, latency=None):
        """
        Calculate GB/s for memory bandwidth bound operations.

        Memory bandwidth calculation:
        - Input activation: M * K * activation_bytes
        - Quantized weights: K * N * w_nbits / 8 bytes
        - Scales: N * (K/group_size) * scale_bytes
        - Output: M * N * output_bytes

        Note: For ref benchmark, we measure the total memory traffic
        including reading input, weights, and writing output.
        """
        x = args[0]
        W_q = args[1]
        m = x.shape[0]
        k = x.shape[1]
        n = W_q.shape[0]

        if self.op_name.startswith("qcgem_w"):
            # QC-GEM: Quantized weights
            # Input activation: M*K*2 bytes (fp16/bf16)
            # Quantized weights: K*N*w_nbits/8 bytes
            # Scales: N*(K/group_size)*2 bytes
            # Output: M*N*2 bytes
            input_bytes = m * k * 2
            weight_bytes = k * n * self.w_nbits / 8
            scale_bytes = n * k // self.group_size * 2
            output_bytes = m * n * 2
            total_bytes = input_bytes + weight_bytes + scale_bytes + output_bytes
        elif self.op_name == "qcgem_ref":
            # Reference GEMM: All FP16/BF16
            # Input: M*K*2, Weight: K*N*2, Output: M*N*2
            input_bytes = m * k * 2
            weight_bytes = k * n * 2  # Weight stored as (N, K), so K*N*2
            output_bytes = m * n * 2
            total_bytes = input_bytes + weight_bytes + output_bytes
        else:
            total_bytes = (m * k + k * n + m * n) * 2

        if latency is not None and latency > 0:
            return total_bytes / latency / 1e9  # GB/s
        return 0

    def get_memory_breakdown(self, args):
        """
        Get detailed memory breakdown for analysis.
        Returns a dictionary with individual memory components.

        Memory breakdown:
        - input_mem: Input activation memory (MB)
        - weight_mem: Weight memory (MB)
        - scale_mem: Scale matrix memory (MB)
        - output_mem: Output memory (MB)
        - total_mem: Total memory (MB)
        - weight_saving: Weight memory saving vs FP16 (%)
        """
        x = args[0]
        W_q = args[1]
        m = x.shape[0]
        k = x.shape[1]
        n = W_q.shape[0]

        if self.op_name.startswith("qcgem_w"):
            # QC-GEM: Quantized weights
            input_bytes = m * k * 2
            weight_bytes = k * n * self.w_nbits / 8
            scale_bytes = n * k // self.group_size * 2
            output_bytes = m * n * 2
            total_bytes = input_bytes + weight_bytes + scale_bytes + output_bytes

            # Calculate FP16 equivalent weight for comparison
            fp16_weight_bytes = k * n * 2

            return {
                "input_mem": input_bytes / (1024 ** 2),
                "weight_mem": weight_bytes / (1024 ** 2),
                "scale_mem": scale_bytes / (1024 ** 2),
                "output_mem": output_bytes / (1024 ** 2),
                "total_mem": total_bytes / (1024 ** 2),
                "fp16_weight_mem": fp16_weight_bytes / (1024 ** 2),
                "weight_saving_pct": (1 - weight_bytes / fp16_weight_bytes) * 100,
                "weight_to_total_ratio": weight_bytes / total_bytes * 100,
            }
        elif self.op_name == "qgem_ref":
            # Reference GEMM: All FP16/BF16
            input_bytes = m * k * 2
            weight_bytes = k * n * 2
            output_bytes = m * n * 2
            total_bytes = input_bytes + weight_bytes + output_bytes

            return {
                "input_mem": input_bytes / (1024 ** 2),
                "weight_mem": weight_bytes / (1024 ** 2),
                "scale_mem": 0,
                "output_mem": output_bytes / (1024 ** 2),
                "total_mem": total_bytes / (1024 ** 2),
                "fp16_weight_mem": weight_bytes / (1024 ** 2),
                "weight_saving_pct": 0,
                "weight_to_total_ratio": weight_bytes / total_bytes * 100,
            }
        else:
            total_bytes = (m * k + k * n + m * n) * 2
            return {
                "input_mem": m * k * 2 / (1024 ** 2),
                "weight_mem": k * n * 2 / (1024 ** 2),
                "scale_mem": 0,
                "output_mem": m * n * 2 / (1024 ** 2),
                "total_mem": total_bytes / (1024 ** 2),
                "fp16_weight_mem": k * n * 2 / (1024 ** 2),
                "weight_saving_pct": 0,
                "weight_to_total_ratio": k * n * 2 / total_bytes * 100,
            }

    def format_memory_breakdown(self, args):
        """Format memory breakdown as a string for display."""
        mem = self.get_memory_breakdown(args)
        return (
            f"Mem[In:{mem['input_mem']:.2f}MB, "
            f"W:{mem['weight_mem']:.2f}MB({mem['weight_saving_pct']:.0f}%↓), "
            f"S:{mem['scale_mem']:.4f}MB, "
            f"Out:{mem['output_mem']:.2f}MB, "
            f"Total:{mem['total_mem']:.2f}MB]"
        )


def qcgem_input_fn(b, m, n, k, cur_dtype, device, w_nbits, group_size, quant_config):
    """
    Generate inputs for QC-GEM benchmark with weight caching.

    Shape semantics (from YAML shape_desc: "B, M, N, K"):
    - b: batch size
    - m: sequence length (M) = S * b
    - n: output dimension (N) = intermediate_dim
    - k: input dimension (K) = hidden_dim

    For MoE FFN shapes [B, S, H, Inter]:
    - x shape: (M, K) = (b*s, h)
    - W_q shape: (N, K) = (inter, h)
    - scales shape: (N, n_groups) = (n, k // group_size)
    - zeros shape: (N, n_groups) = (n, k // group_size)

    GEMM operation: y = x @ W_q.t()
    where x is (M, K), W_q is (N, K), output y is (M, N)

    The benchmark flow is:
    1. Load pre-cached quantized weights (W_q, scales) from disk
    2. Generate random input activations x
    3. Run qcgem_mm kernel
    4. Return output

    NOTE: Quantization is done offline and cached. The benchmark
    starts from loading cached quantized weights.
    """
    weight_manager = get_weight_manager()

    # Input activation: (M, K) = (b*s, h)
    # Float8 doesn't support torch.randn, so generate in fp16 then convert
    if cur_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        inp = torch.randn([m, k], dtype=torch.float16, device=device).to(cur_dtype)
    else:
        inp = torch.randn([m, k], dtype=cur_dtype, device=device)

    # Get or create quantized weights
    # Weight shape is (N, K) = (inter, h)
    cache_key = f"benchmark_{quant_config}_{b}_{m}_{n}_{k}"

    # Try to load from cache first
    W_q, scales = None, None
    try:
        cached = weight_manager.quantizer.load_quantized_weights(cache_key, quant_config)
        if cached is not None:
            W_q, scales = cached
            W_q = W_q.to(device)
            scales = scales.to(device)
    except Exception:
        pass

    if W_q is None:
        # Need to create and cache weights
        # For benchmark, we use a fixed seed for reproducibility
        seed = hash(f"{quant_config}_{b}_{m}_{n}_{k}") % (2**32)

        # Create float weights in (N, K) format
        if cur_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            W_float = torch.randn(n, k, device=device, dtype=torch.float16)
            W_float = W_float.to(cur_dtype)
        else:
            W_float = torch.randn(n, k, device=device, dtype=cur_dtype)

        # Quantize and cache
        W_q, scales, _ = weight_manager.quantizer.save_quantized_weights(
            W_float.cpu(), quant_config, model_name=cache_key
        )
        W_q = W_q.to(device)
        scales = scales.to(device)

    # Zeros - shape (N, n_groups) = (n, k // group_size)
    if cur_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        zeros = torch.zeros(n, k // group_size, device=device, dtype=torch.float16)
    else:
        zeros = torch.zeros(n, k // group_size, device=device, dtype=cur_dtype)

    yield inp, W_q, scales, zeros, w_nbits, group_size, cur_dtype


def qcgem_ref_input_fn(b, m, n, k, cur_dtype, device, w_nbits, group_size, quant_config):
    """
    Generate inputs for reference GEMM (dequantized weights).

    Shape semantics same as qcgem_input_fn.
    For MoE FFN shapes [B, S, H, Inter]:
    - x shape: (M, K) = (b*s, h)
    - W shape: (N, K) = (inter, h)
    - Result: (M, N)

    GEMM operation: y = x @ W.t()
    where x is (M, K), W is (N, K), output y is (M, N)

    The benchmark flow for reference GEMM is:
    1. Load float weights (FP16/BF16) from memory
    2. Generate random input activations x
    3. Run torch.mm (baseline) or flag_gems.mm
    4. Return output

    NOTE: To ensure fair comparison, this function uses the SAME weight shape
    as qcgem_input_fn: (N, K) = (intermediate_dim, hidden_dim).
    The weight is NOT transposed, matching the quantized version.
    """
    # Input: (M, K) = (b*s, h)
    if cur_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        inp = torch.randn([m, k], dtype=torch.float16, device=device).to(cur_dtype)
    else:
        inp = torch.randn([m, k], dtype=cur_dtype, device=device)

    # Create float weights (N, K) = (inter, h) format
    # Use same seed as qcgem_input_fn for fair comparison
    seed = hash(f"{quant_config}_{b}_{m}_{n}_{k}") % (2**32)
    torch.manual_seed(seed)

    if cur_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        W_float = torch.randn(n, k, device=device, dtype=torch.float16)
        W_float = W_float.to(cur_dtype)
    else:
        W_float = torch.randn(n, k, device=device, dtype=cur_dtype)

    # DO NOT transpose - keep as (N, K) to match quantized version
    # GEMM will compute: y = x @ W.t() internally
    W_deq = W_float

    yield inp, W_deq


def qcgem_op(inp, W_q, scales, zeros, w_nbits, group_size, input_dtype):
    """QC-GEM operation using flag_gems."""
    from flag_gems.ops.qcgem import qcgem_mm, DType
    from flag_gems.ops.qcgem.dtypes import TORCH_TO_DTYPE

    qcgem_dtype = TORCH_TO_DTYPE.get(input_dtype, DType.FP16)
    return qcgem_mm(inp, W_q, scales, zeros, w_nbits, group_size, qcgem_dtype)


def qcgem_ref_op(inp, W_deq):
    """
    Reference GEMM with dequantized weights.

    GEMM: y = x @ W.t()
    where inp is (M, K), W_deq is (N, K), output is (M, N)
    """
    # W_deq is (N, K), so W_deq.t() is (K, N)
    # inp @ W_deq.t() = (M, K) @ (K, N) = (M, N)
    return torch.mm(inp, W_deq.t())


def qcgem_ref_gems_op(inp, W_deq):
    """
    Reference GEMM using flag_gems.mm.
    Same as torch_op but uses FlagGems for fair comparison.
    """
    return torch.mm(inp, W_deq.t())


@pytest.mark.qcgem_w8a16
def test_qcgem_w8a16_benchmark():
    """
    Benchmark QC-GEM w8A16: 8-bit weight, 16-bit activation (FP16/BF16).
    Usage:
        pytest -s ./benchmark/test_qcgem_perf.py::test_qcgem_w8a16_benchmark \
            -m qcgem_w8a16 --shape_file models_shapes/Qwen3.5-397B-A17B-p32768d1024.yaml \
            --mode kernel --dtypes float16,bfloat16
    """
    config = QUANT_CONFIGS["w8A16"]
    bench_qcgem = QCGEMBenchmark(
        input_fn=qcgem_input_fn,
        op_name="qcgem_w8a16",
        torch_op=qcgem_op,
        dtypes=[torch.float16, torch.bfloat16],
        w_nbits=config["w_nbits"],
        group_size=config["group_size"],
        quant_config="w8A16",
    )
    bench_qcgem.run()


@pytest.mark.qcgem_w4a16
def test_qcgem_w4a16_benchmark():
    """
    Benchmark QC-GEM w4A16: 4-bit weight, 16-bit activation (FP16/BF16).
    Usage:
        pytest -s ./benchmark/test_qcgem_perf.py::test_qcgem_w4a16_benchmark \
            -m qcgem_w4a16 --shape_file models_shapes/Qwen3.5-397B-A17B-p32768d1024.yaml \
            --mode kernel --dtypes float16,bfloat16
    """
    config = QUANT_CONFIGS["w4A16"]
    bench_qcgem = QCGEMBenchmark(
        input_fn=qcgem_input_fn,
        op_name="qcgem_w4a16",
        torch_op=qcgem_op,
        dtypes=[torch.float16, torch.bfloat16],
        w_nbits=config["w_nbits"],
        group_size=config["group_size"],
        quant_config="w4A16",
    )
    bench_qcgem.run()


@pytest.mark.qcgem_w8a8
def test_qcgem_w8a8_benchmark():
    """
    Benchmark QC-GEM w8A8: 8-bit weight, 8-bit activation (INT8-INT8).
    权重使用INT8量化，激活值使用INT8量化，计算内核采用INT8计算，输出FP16矩阵。
    Usage:
        pytest -s ./benchmark/test_qcgem_perf.py::test_qcgem_w8a8_benchmark \
            -m qcgem_w8a8 --shape_file models_shapes/Qwen3.5-397B-A17B-p32768d1024.yaml \
            --mode kernel --dtypes float16
    """
    config = QUANT_CONFIGS["w8A8"]
    bench_qcgem = QCGEMBenchmark(
        input_fn=qcgem_input_fn,
        op_name="qcgem_w8a8",
        torch_op=qcgem_op,
        dtypes=[torch.float16],  # 输出类型为FP16，输入激活值在kernel内部量化
        w_nbits=config["w_nbits"],
        group_size=config["group_size"],
        quant_config="w8A8",
    )
    bench_qcgem.run()


@pytest.mark.qcgem_ref
def test_qcgem_ref_benchmark():
    """
    Benchmark reference GEMM with dequantized weights.

    NOTE: This benchmark uses the SAME shapes as quantized versions
    for fair memory comparison. Shape semantics: (B, M, N, K)
    where:
    - M = batch * seq_len (total tokens)
    - N = intermediate_dim (output features)
    - K = hidden_dim (input features)

    Usage:
        pytest -s ./benchmark/test_qcgem_perf.py::test_qcgem_ref_benchmark \
            -m qcgem_ref --shape_file models_shapes/Qwen3.5-397B-A17B-p32768d1024.yaml \
            --mode kernel --dtypes float16,bfloat16
    """
    bench_ref = QCGEMBenchmark(
        input_fn=qcgem_ref_input_fn,
        op_name="qcgem_ref",
        torch_op=qcgem_ref_op,
        gems_op=qcgem_ref_gems_op,
        dtypes=[torch.float16, torch.bfloat16],
        w_nbits=4,  # Same as w4A16 for consistency
        group_size=128,
        quant_config="w4A16",  # Use same config for cache key matching
    )
    bench_ref.run()
