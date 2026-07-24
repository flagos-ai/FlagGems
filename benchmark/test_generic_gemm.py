# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flag_gems.ops.generic_gemm import generic_gemm
from flag_gems.utils.device_info import get_device_capability

from . import base, consts

try:
    from transformer_engine.pytorch.cpp_extensions.gemm import (
        general_gemm as te_general_gemm,
    )

    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False


def _torch_gelu(x):
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + torch.tanh(inner))


def torch_ref(inp, weight, *, bias=None, gelu=False):
    out = torch.mm(inp, weight.T)
    if bias is not None:
        out = out + bias
    pre_gelu = out.clone() if gelu else None
    if gelu:
        out = _torch_gelu(out)
    return out, None, pre_gelu, None


if TE_AVAILABLE:
    _baseline_op = lambda inp, weight, **kw: te_general_gemm(weight, inp, **kw)
else:
    _baseline_op = torch_ref
_gems_op = lambda inp, weight, **kw: generic_gemm(inp, weight, layout="NT", **kw)


FP8_SUPPORTED = get_device_capability() >= (9, 0)
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


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


if TE_AVAILABLE:
    from transformer_engine.pytorch import fp8_autocast


def _per_tensor_quantize(x: torch.Tensor):
    amax = x.abs().max()
    scale = (amax / FP8_MAX).to(torch.float32)
    x_fp8 = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, scale


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
    inp_bf16 = torch.randn([m, k], dtype=torch.bfloat16, device=device)
    weight_bf16 = torch.randn([n, k], dtype=torch.bfloat16, device=device)
    yield inp_bf16, weight_bf16, {}


def _gems_fp8_op(inp, weight, **kw):
    from flag_gems.ops.generic_gemm import _cached_per_tensor_quantize

    inp_fp8, scale_a = _cached_per_tensor_quantize(inp)
    weight_fp8, scale_b = _cached_per_tensor_quantize(weight)
    return generic_gemm(
        inp_fp8, weight_fp8, layout="NT", scale_a=scale_a, scale_b=scale_b
    )


def _torch_fp8_baseline(inp, weight, **kw):
    amax_a = inp.abs().max()
    scale_a = (amax_a / FP8_MAX).to(torch.float32)
    a_q = (inp.float() / scale_a).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

    amax_b = weight.abs().max()
    scale_b = (amax_b / FP8_MAX).to(torch.float32)
    b_q = (weight.float() / scale_b).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

    out = a_q.float() @ b_q.float().T
    out = (out * scale_a * scale_b).to(torch.bfloat16)
    return out, None, None, None


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
        torch_op=_fp8_baseline,
        gems_op=_gems_fp8_op,
        dtypes=[torch.bfloat16],
    )
    bench.run()
