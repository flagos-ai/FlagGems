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

import os

import pytest
import torch
import triton
import yaml

import flag_gems

from . import base, consts
from .conftest import Config


def mm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.t()
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mm():
    bench = base.BlasBenchmark(
        op_name="mm",
        input_fn=mm_input_fn,
        torch_op=torch.Tensor.mm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


_MM_W8A8_FP8_OUT_CACHE = {}
_MM_W8A8_FP8_OUT_CACHE_MAX_ENTRIES = 8


def _mm_w8a8_fp8_out_cached(a, b):
    out_dtype = torch.bfloat16
    device_index = a.device.index if a.device.index is not None else -1
    key = (device_index, a.shape[0], b.shape[1], out_dtype)
    out = _MM_W8A8_FP8_OUT_CACHE.get(key)
    if out is None or out.device != a.device:
        out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype)
        _MM_W8A8_FP8_OUT_CACHE[key] = out
        while len(_MM_W8A8_FP8_OUT_CACHE) > _MM_W8A8_FP8_OUT_CACHE_MAX_ENTRIES:
            _MM_W8A8_FP8_OUT_CACHE.pop(next(iter(_MM_W8A8_FP8_OUT_CACHE)))
    else:
        _MM_W8A8_FP8_OUT_CACHE.pop(key)
        _MM_W8A8_FP8_OUT_CACHE[key] = out
    return flag_gems.mm_w8a8_fp8_out(a, b, out=out)


def mm_w8a8_fp8_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    for a, weight in mm_input_fn(b, m, n, k, torch.float32, device, b_column_major):
        yield a.to(cur_dtype), weight.to(cur_dtype)


class MmW8A8Fp8Benchmark(base.BlasBenchmark):
    def get_input_iter(self, dtype):
        # vLLM CUTLASS expects row-major A and column-major B. Both paths use
        # these same prequantized tensors; preparation is outside timing.
        for b, m, n, k in self.shapes:
            yield from self.input_fn(b, m, n, k, dtype, self.device, True)

    def get_latency(self, op, *args, **kwargs):
        if op is not self.torch_op:
            # Populate descriptor, output, and autotune caches before capture.
            for _ in range(2):
                op(*args, **kwargs)
            torch.cuda.synchronize()
        return triton.testing.do_bench_cudagraph(
            lambda: op(*args, **kwargs),
            rep=Config.repetition,
            return_mode="median",
        )

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        if not shape_file_path or not os.path.isfile(shape_file_path):
            return
        with open(shape_file_path, "r", encoding="utf-8") as shape_file:
            yaml_config = yaml.safe_load(shape_file) or {}
        if "mm" not in yaml_config:
            return
        self.shapes = [
            tuple(shape)
            for shape in yaml_config["mm"].get("shapes", self.DEFAULT_SHAPES)
        ]
        self.shape_desc = yaml_config["mm"].get("shape_desc", self.shape_desc)

    def get_tflops(self, op, *args, **kwargs):
        return args[0].shape[0] * args[0].shape[1] * args[1].shape[1] * 2


@pytest.mark.mm_w8a8_fp8
def test_mm_w8a8_fp8():
    if not hasattr(flag_gems, "mm_w8a8_fp8_out"):
        pytest.skip("mm_w8a8_fp8 benchmark requires the Hopper W8A8 backend")
    vllm_ops = pytest.importorskip("vllm._custom_ops")
    scale = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

    def vllm_fp8_mm(a, b):
        return vllm_ops.cutlass_scaled_mm(a, b, scale, scale, torch.bfloat16)

    bench = MmW8A8Fp8Benchmark(
        input_fn=mm_w8a8_fp8_input_fn,
        op_name="mm_w8a8_fp8",
        torch_op=vllm_fp8_mm,
        dtypes=consts.FP8_DTYPES,
    )
    bench.set_gems(_mm_w8a8_fp8_out_cached)
    bench.run()


class MmSelfTransposeBenchmark(base.GenericBenchmark2DOnly):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_more_shapes(self):
        return []

    def get_tflops(self, op, *args, **kwargs):
        m, k = args[0].shape
        return 2 * m * m * k


def _input_fn(shape, cur_dtype, device):
    m, k = shape
    inp = torch.randn([k, m], dtype=cur_dtype, device=device).t()

    yield inp,


def torch_mm_self_transpose(inp):
    return torch.mm(inp, inp.t())


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mm_self_transpose_benchmark():
    bench = MmSelfTransposeBenchmark(
        op_name="mm_self_transpose",
        input_fn=_input_fn,
        torch_op=torch_mm_self_transpose,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


def mm_out_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        out = torch.empty([m, n], dtype=cur_dtype, device=device)
        yield inp1, inp2.t(), {"out": out}
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        out = torch.empty([m, n], dtype=cur_dtype, device=device)
        yield inp1, inp2, {"out": out}


@pytest.mark.mm_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mm_out():
    bench = base.BlasBenchmark(
        op_name="mm_out",
        input_fn=mm_out_input_fn,
        torch_op=torch.mm,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
