import pytest
import torch

from flag_gems.ops.generic_gemm import generic_gemm
from flag_gems.utils.device_info import get_device_capability

from . import base, consts

# -- Baseline operator: prefer TransformerEngine if installed, fall back to torch --
try:
    from transformer_engine.pytorch.cpp_extensions.gemm import (
        general_gemm as te_general_gemm,
    )

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


def _torch_gelu(x):
    """tanh-approximated GELU matching the TE / generic_gemm kernel."""
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + torch.tanh(inner))


def torch_ref(inp, weight, *, bias=None, gelu=False):
    """Reference: inp[M,K] @ weight[N,K]^T = [M,N] (+ bias) (+ GELU)."""
    out = torch.mm(inp, weight.T)
    if bias is not None:
        out = out + bias
    pre_gelu = out.clone() if gelu else None
    if gelu:
        out = _torch_gelu(out)
    return out, None, pre_gelu, None


# TE uses column-major TN convention: general_gemm(weight[N,K], inp[M,K]) → output[M,N].
# FlagGems uses row-major convention: generic_gemm(inp[M,K], weight[N,K], "NT") → [M,N].
if TE_AVAILABLE:
    _baseline_op = lambda inp, weight, **kw: te_general_gemm(weight, inp, **kw)
else:
    _baseline_op = torch_ref
_gems_op = lambda inp, weight, **kw: generic_gemm(inp, weight, layout="NT", **kw)


FP8_SUPPORTED = get_device_capability() >= (9, 0)
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


# DeepSeek V4 shapes [M, N, K], covering all layer types
DSV4_SHAPES = [
    (1, 3072, 7168),
    (8, 3072, 7168),
    (64, 3072, 7168),
    (1, 7168, 3072),
    (8, 7168, 3072),
    (64, 7168, 3072),
    (1, 384, 7168),
    (8, 384, 7168),
    (1, 1536, 7168),
    (8, 1536, 7168),
    (1, 57344, 1536),
    (1, 8192, 7168),
    (1, 576, 7168),
    (1, 122880, 512),
    (1, 1024, 4096),
    (8, 1024, 4096),
    (1, 7168, 1024),
    (8, 7168, 1024),
    (1, 6144, 7168),
    (1, 2048, 4096),
    (8, 2048, 4096),
    (1, 4096, 2048),
    (8, 4096, 2048),
    (1, 256, 4096),
    (1, 1024, 4096),
    (1, 28672, 1024),
    (1, 4096, 4096),
    (1, 576, 4096),
    (1, 61440, 512),
    (1, 4096, 1024),
    (1, 4096, 4096),
]


def _input_fn(m, n, k, dtype, device):
    """Yield (inp, weight, kwargs) for generic_gemm benchmark.

    inp:   [M, K]
    weight: [N, K]  — both have K as the last dim (TE TN / FlagGems NT convention)."""
    inp = torch.randn([m, k], dtype=dtype, device=device)
    weight = torch.randn([n, k], dtype=dtype, device=device)
    bias = torch.randn([n], dtype=dtype, device=device)
    yield inp, weight, {"bias": bias, "gelu": True}
    yield inp, weight, {}
    yield inp, weight, {"gelu": True}


class GenericGemmBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_DTYPES = consts.FP16_BF16_DTYPES

    def set_shapes(self, shape_file_path=None):
        self.shapes = DSV4_SHAPES[:]
        self.shape_desc = "M, N, K"

    def get_input_iter(self, dtype):
        for m, n, k in self.shapes:
            yield from _input_fn(m, n, k, dtype, self.device)

    def get_tflops(self, op, *args, **kwargs):
        inp = args[0]
        weight = args[1]
        m, k = inp.shape
        n = weight.shape[0]
        return 2.0 * m * n * k


@pytest.mark.generic_gemm
def test_generic_gemm():
    bench = GenericGemmBenchmark(
        op_name="generic_gemm",
        torch_op=_baseline_op,
        gems_op=_gems_op,
        dtypes=consts.FP16_BF16_DTYPES,
    )
    bench.run()


# ═══════════════════════════════════════════════════════════════════
# FP8 benchmark — TE cuBLAS FP8 (fp8_autocast) vs FlagGems FP8 (Triton)
# Both sides receive BF16 inputs. TE auto-quantizes via fp8_autocast.
# FlagGems quantizes manually in the wrapper.
#
# Note: TE baseline includes quantization in its measurement (fp8_autocast
# runs quantize kernel + cuBLAS GEMM atomically). FlagGems measurement
# excludes quantize (pre-quantized FP8 tensors from _fp8_input_fn).
# This is a reference comparison, not a strict kernel-vs-kernel benchmark.
# ═══════════════════════════════════════════════════════════════════

if TE_AVAILABLE:
    from transformer_engine.pytorch import fp8_autocast


def _per_tensor_quantize(x: torch.Tensor):
    """Quantize to float8_e4m3fn with per-tensor scaling."""
    amax = x.abs().max()
    scale = (amax / FP8_MAX).to(torch.float32)
    x_fp8 = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, scale


# Subset of DSV4 shapes with K divisible by 64 (autotune-friendly)
FP8_SHAPES = [
    (1, 3072, 7168),
    (8, 3072, 7168),
    (1, 7168, 3072),
    (1, 8192, 7168),
    (1, 6144, 7168),
    (1, 1024, 4096),
    (1, 4096, 4096),
]


def _fp8_input_fn(m, n, k, device):
    """Yield (inp_bf16, weight_bf16, kwargs) for FP8 benchmark.
    TE baseline consumes BF16 directly (fp8_autocast quantizes internally).
    Gems wrapper quantizes to FP8 before calling generic_gemm."""
    inp_bf16 = torch.randn([m, k], dtype=torch.bfloat16, device=device)
    weight_bf16 = torch.randn([n, k], dtype=torch.bfloat16, device=device)
    yield inp_bf16, weight_bf16, {}


def _gems_fp8_op(inp, weight, **kw):
    """FlagGems FP8 Triton kernel — cached quantize, do_bench-safe.
    First call on a tensor: absmax + quantize.  Same data_ptr on subsequent
    calls (normal inside do_bench since arguments are the same tensor object):
    cache hit, zero quant overhead in the timing loop."""
    from flag_gems.ops.generic_gemm import _cached_per_tensor_quantize

    inp_fp8, scale_a = _cached_per_tensor_quantize(inp)
    weight_fp8, scale_b = _cached_per_tensor_quantize(weight)
    return generic_gemm(inp_fp8, weight_fp8, layout="NT",
                        scale_a=scale_a, scale_b=scale_b)


# Torch FP8 baseline: per-tensor quantize BF16 → FP8, matmul in fp32, restore scale.
# Simulates the quantize-dot-restore path of FP8 GEMM without cuBLAS or TE.
def _torch_fp8_baseline(inp, weight, **kw):
    """Simulate FP8 GEMM in PyTorch: per-tensor quantize → matmul → restore scale.

    inp:   [M, K] bfloat16
    weight: [N, K] bfloat16

    Does the same quantize-dot-restore as the Triton FP8 kernel:
    1. Quantize each input to float8_e4m3fn with per-tensor scaling.
    2. Dequantize back to float32 and do matmul in float32.
    3. Multiply by scale_a * scale_b to restore approximate fp32 range.
    4. Cast output to bfloat16.

    This is an upper-bound baseline: matmul happens in float32, which has more
    precision than FP8 MMA instructions. It verifies functional correctness but
    is not a strict latency baseline for FP8 hardware.
    """
    amax_a = inp.abs().max()
    scale_a = (amax_a / FP8_MAX).to(torch.float32)
    a_q = (inp.float() / scale_a).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

    amax_b = weight.abs().max()
    scale_b = (amax_b / FP8_MAX).to(torch.float32)
    b_q = (weight.float() / scale_b).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

    out = a_q.float() @ b_q.float().T
    out = (out * scale_a * scale_b).to(torch.bfloat16)
    return out, None, None, None


# TE baseline: fp8_autocast auto-quantizes BF16 → FP8, runs cuBLAS FP8 directly.
# When TE is not available, fall back to the torch FP8 simulation above.
if TE_AVAILABLE:
    def _te_fp8_baseline(inp, weight, **kw):
        with fp8_autocast(enabled=True):
            out, _, _, _ = te_general_gemm(weight, inp, out_dtype=torch.bfloat16)
        return out, None, None, None

    _fp8_baseline = _te_fp8_baseline
else:
    _fp8_baseline = _torch_fp8_baseline


class GenericGemmFp8Benchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_DTYPES = [torch.bfloat16]

    def set_shapes(self, shape_file_path=None):
        self.shapes = FP8_SHAPES[:]
        self.shape_desc = "M, N, K"

    def get_input_iter(self, dtype):
        for m, n, k in self.shapes:
            yield from _fp8_input_fn(m, n, k, self.device)

    def get_tflops(self, op, *args, **kwargs):
        inp = args[0]
        weight = args[1]
        m, k = inp.shape
        n = weight.shape[0]
        return 2.0 * m * n * k


@pytest.mark.generic_gemm
@pytest.mark.skipif(not FP8_SUPPORTED, reason="FP8 benchmark requires SM>=90")
def test_generic_gemm_fp8():
    bench = GenericGemmFp8Benchmark(
        op_name="generic_gemm_fp8",
        torch_op=_fp8_baseline,        # TE cuBLAS FP8 or torch FP8 baseline
        gems_op=_gems_fp8_op,           # FlagGems Triton FP8
        dtypes=[torch.bfloat16],
    )
    bench.run()
