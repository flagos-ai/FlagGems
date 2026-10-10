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

import statistics

import pytest
import torch

import flag_gems

from . import consts
from .conftest import Config
from .consts import FLOAT_DTYPES, BenchmarkMetrics
from .test_blas_perf_parallel import (
    ParallelBlasBenchmark,
    ParallelMmW8A8Fp8Benchmark,
    _mm_w8a8_fp8_output_dtype,
    mm_input_fn,
)


class ParallelMmW8A8Int8Benchmark(ParallelBlasBenchmark):
    """W8A8 INT8 workloads from prequantized INT8 inputs.

    Use --mode cudagraph --level comprehensive --dtypes bfloat16 --dtypes
    float16 to cover the reference PR's 121 configurations and two B layouts.
    The Torch baseline uses BF16, independently of the original input dtype.
    """

    SHAPE_CONFIG_KEYS = ("mm_w8a8_int8", "BlasBenchmark")

    def prepare_call(self, op, a, b):
        if op is self.torch_op:
            a_bf16, b_bf16 = a.to(torch.bfloat16), b.to(torch.bfloat16)
            return lambda: op(a_bf16, b_bf16)
        # Quantize both inputs offline; time only the public scaled INT8 GEMM.
        activation = a.float()
        peak_a = activation.abs().amax(dim=1).clamp_min(1e-10)
        scale_a = peak_a * (1.0 / 127.0)
        aq = (
            (activation / peak_a[:, None] * 127.0)
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
        )
        weight = b.float()
        peak = weight.abs().amax(dim=0).clamp_min(1e-10)
        scale_b = peak * (1.0 / 127.0)
        bq = (weight / peak[None, :] * 127.0).round().clamp(-127, 127).to(torch.int8)
        bq = bq.t().contiguous().t()
        out = torch.empty(
            (a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16
        )
        return lambda: flag_gems.mm_w8a8_int8_out(aq, bq, scale_a, scale_b, out=out)

    def get_latency(self, op, *args, **kwargs):
        # Input quantization, layouts, baseline casts and output allocation are offline.
        # Time INT8 GEMM, scaling and any necessary reduction kernels.
        return super().get_latency(self.prepare_call(op, *args), **kwargs)

    def get_tflops(self, op, *args, **kwargs):
        a, b = args
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]


class AscendMmW8A8Int8Benchmark(ParallelMmW8A8Int8Benchmark):
    """Prequantized inputs and paired public-call timing against INT8 or BF16."""

    def __init__(self, *args, baseline, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline = baseline

    def get_input_iter(self, cur_dtype):
        for _, m, n, k in self.shapes:
            torch.manual_seed(0)
            a = torch.randint(-128, 128, (m, k), device=self.device, dtype=torch.int8)
            b = torch.randint(
                -128, 128, (n, k), device=self.device, dtype=torch.int8
            ).t()
            sa = torch.rand(m, device=self.device) * 0.01
            sb = torch.rand(n, device=self.device) * 0.01
            yield a, b, sa, sb

    def get_parallel_metric_group_size(self, shape):
        return 1

    def _time_callable(self, fn, xs):
        # Ascend has no triton do_bench_cudagraph, so NPU graph timing lives here
        # in the operator benchmark instead of the shared benchmark base.
        if Config.mode != consts.BenchMode.CUDAGRAPH:
            return super()._time_callable(fn, xs)
        torch_device_fn = flag_gems.runtime.torch_device_fn
        stream = torch_device_fn.Stream()
        stream.wait_stream(torch_device_fn.current_stream())
        with torch_device_fn.stream(stream):
            fn()
            start = torch_device_fn.Event(enable_timing=True)
            end = torch_device_fn.Event(enable_timing=True)
            start.record()
            for _ in range(5):
                fn()
            end.record()
            torch_device_fn.synchronize()
            estimate = start.elapsed_time(end) / 5
            repeats = (
                1000 if estimate == 0 else max(1, int(Config.repetition / estimate))
            )
            graph = torch_device_fn.NPUGraph()
            with torch_device_fn.graph(graph, stream=stream):
                for _ in range(repeats):
                    fn()
            torch_device_fn.synchronize()
            samples = []
            for _ in range(10):
                start = torch_device_fn.Event(enable_timing=True)
                end = torch_device_fn.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                torch_device_fn.synchronize()
                samples.append(start.elapsed_time(end) / repeats)
            graph.reset()
        torch_device_fn.current_stream().wait_stream(stream)
        return statistics.median(samples)

    def _build_metric_from_input(self, input_item):
        import torch_npu

        a, b, sa, sb = input_item
        out = torch.empty(
            (a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16
        )

        def candidate():
            return flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)

        if self.baseline == "native_int8":

            def baseline():
                return torch_npu.npu_quant_matmul(
                    a, b, sb, pertoken_scale=sa, output_dtype=torch.bfloat16
                )

        else:
            a_bf16 = (a.float() * sa[:, None]).bfloat16()
            b_bf16 = (b.t().float() * sb[:, None]).bfloat16().t()
            out_bf16 = torch.empty_like(out)

            def baseline():
                return torch.mm(a_bf16, b_bf16, out=out_bf16)

        calls = (candidate, baseline)
        timings = ([], [])
        for round_index in range(3):
            for index in (0, 1) if round_index % 2 == 0 else (1, 0):
                latency = self._time_callable(calls[index], None)
                timings[index].append(latency)
        latency, latency_base = map(statistics.median, timings)
        return BenchmarkMetrics(
            shape_detail=self.record_shapes(a, b),
            latency=latency,
            latency_base=latency_base,
            speedup=latency_base / latency,
            tflops=2 * a.shape[0] * b.shape[1] * a.shape[1] / latency / 1e9,
        )


@pytest.mark.mm_w8a8_int8
@pytest.mark.parametrize(
    "baseline",
    ["bf16", "native_int8"] if flag_gems.vendor_name == "ascend" else ["bf16"],
)
def test_mm_w8a8_int8(baseline):
    if flag_gems.vendor_name == "thead" or not hasattr(flag_gems, "mm_w8a8_int8_out"):
        pytest.skip("mm_w8a8_int8 is not implemented by the active backend")
    bench_cls = (
        AscendMmW8A8Int8Benchmark
        if flag_gems.vendor_name == "ascend"
        else ParallelMmW8A8Int8Benchmark
    )
    options = {"baseline": baseline} if flag_gems.vendor_name == "ascend" else {}
    bench = bench_cls(
        input_fn=mm_input_fn,
        op_name="mm_w8a8_int8",
        torch_op=torch.mm,
        dtypes=[torch.bfloat16] if flag_gems.vendor_name == "ascend" else FLOAT_DTYPES,
        **options,
    )
    bench.set_gems(flag_gems.mm_w8a8_int8)
    print(f"Baseline: {baseline}; prequantized INT8 GEMM and scaling timed")
    bench.run()


class ParallelMmW8A8Int8THeadBenchmark(ParallelMmW8A8Fp8Benchmark):
    """Reuse upstream workloads and timing with prequantized INT8 inputs."""

    def get_latency(self, op, *args, **kwargs):
        if op is not self.torch_op:
            a, b = args
            out_dtype = _mm_w8a8_fp8_output_dtype(a)
            if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(
                    "INT8 W8A8 benchmark requires a floating non-FP8 output"
                )
            peak_a = a.float().abs().amax(dim=1).clamp_min(1e-10)
            scale_a = peak_a * (1.0 / 127.0)
            a_q = (
                torch.round(a.float() / peak_a[:, None] * 127)
                .clamp(-127, 127)
                .to(torch.int8)
            )
            peak_b = b.float().abs().amax(dim=0).clamp_min(1e-10)
            scale_b = peak_b * (1.0 / 127.0)
            b_q = (
                torch.round((b.float() / peak_b[None, :]) * 127)
                .clamp(-127, 127)
                .to(torch.int8)
                .t()
                .contiguous()
                .t()
            )
            out = torch.empty(
                (a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype
            )

            # Both quantizations, scales and the output allocation are untimed.
            # Replay invokes the public scaled-input API, including any copies.
            def op():
                return flag_gems.mm_w8a8_int8_out(a_q, b_q, scale_a, scale_b, out=out)

            args, kwargs = (), {}
        return super().get_latency(op, *args, **kwargs)

    def get_tflops(self, op, *args, **kwargs):
        a, b = args
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]


@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_thead():
    if flag_gems.vendor_name != "thead" or not hasattr(flag_gems, "mm_w8a8_int8_out"):
        pytest.skip("mm_w8a8_int8 benchmark requires the THead INT8 backend")
    bench = ParallelMmW8A8Int8THeadBenchmark(
        input_fn=mm_input_fn,
        op_name="mm_w8a8_int8",
        torch_op=torch.Tensor.mm,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.mm_w8a8_int8)
    bench.run()
