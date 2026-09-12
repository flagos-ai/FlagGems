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
import statistics

import pytest
import torch
import triton
import yaml

import flag_gems
from flag_gems.runtime import torch_device_fn

from . import base, consts
from .conftest import Config

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
    a = torch.randn([m, k], dtype=torch.float32, device=device)
    if b_column_major:
        weight = torch.randn([n, k], dtype=torch.float32, device=device).t()
    else:
        weight = torch.randn([k, n], dtype=torch.float32, device=device)
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
            torch_device_fn.synchronize()
        if flag_gems.vendor_name == "mthreads":
            # Bound captured output allocations for large GEMMs on either path.
            output_bytes = args[0].shape[0] * args[1].shape[1] * 2
            count = max(1, min(32, (128 * 1024 * 1024) // max(output_bytes, 1)))
            stream = torch.musa.Stream()
            stream.wait_stream(torch.musa.current_stream())
            with torch.musa.stream(stream):
                for _ in range(3):
                    op(*args, **kwargs)
                graph = torch.musa.MUSAGraph()
                with torch.musa.graph(graph):
                    for _ in range(count):
                        op(*args, **kwargs)
            torch.musa.current_stream().wait_stream(stream)
            graph.replay()
            torch.musa.synchronize()
            start = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            estimate = max(start.elapsed_time(end), 1.0e-3)
            for _ in range(max(1, int(Config.warm_up / estimate))):
                graph.replay()
            torch.musa.synchronize()
            times = []
            elapsed = 0.0
            while len(times) < 5 or elapsed < Config.repetition:
                start, end = (
                    torch.musa.Event(enable_timing=True),
                    torch.musa.Event(enable_timing=True),
                )
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                ms = start.elapsed_time(end)
                elapsed += ms
                times.append(ms / count)
            return statistics.median(times)
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
        pytest.skip("mm_w8a8_fp8 benchmark requires a supported FP8 backend")
    scale = torch.ones(1, dtype=torch.float32, device=flag_gems.device)
    if flag_gems.vendor_name == "mthreads":
        vllm_ops = pytest.importorskip(
            "vllm.model_executor.layers.quantization.utils.w8a8_utils"
        )

        def vllm_fp8_mm(a, b):
            return vllm_ops.torch_per_tensor_w8a8_scaled_mm(
                qinput=a,
                weight=b,
                out_dtype=torch.bfloat16,
                scale_a=scale,
                scale_b=scale,
                bias=None,
                output_shape=[a.shape[0], b.shape[1]],
            )

    else:
        vllm_ops = pytest.importorskip("vllm._custom_ops")

        def vllm_fp8_mm(a, b):
            return vllm_ops.cutlass_scaled_mm(a, b, scale, scale, torch.bfloat16)

    bench = MmW8A8Fp8Benchmark(
        input_fn=mm_w8a8_fp8_input_fn,
        op_name="mm_w8a8_fp8",
        torch_op=vllm_fp8_mm,
        dtypes=(
            [torch.float8_e4m3fn]
            if flag_gems.vendor_name == "mthreads"
            else consts.FP8_DTYPES
        ),
    )
    bench.set_gems(_mm_w8a8_fp8_out_cached)
    bench.run()
