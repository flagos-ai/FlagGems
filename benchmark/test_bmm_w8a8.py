# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import gc
import json
import os
import statistics
from pathlib import Path

import pytest
import torch

import flag_gems

from . import base
from .test_blas_perf_parallel import ParallelBenchmarkMixin


def make_channel_int8_bmm_inputs(
    shape: tuple[int, int, int, int], device: str, dtype: torch.dtype = torch.int8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, rows, columns, reduction = shape
    torch.manual_seed(0)
    activation = torch.randint(
        -128, 128, (batch, rows, reduction), device=device, dtype=dtype
    )
    weight = torch.randint(
        -128, 128, (batch, columns, reduction), device=device, dtype=dtype
    ).transpose(1, 2)
    activation_scale = torch.rand((batch, rows), device=device) / 127
    weight_scale = torch.rand((batch, columns), device=device) / 127
    return activation, weight, activation_scale, weight_scale


def bf16_bmm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    baseline_activation: torch.Tensor,
    baseline_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(baseline_activation, baseline_weight)


def native_int8_bmm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    baseline_activation: torch.Tensor,
    baseline_weight: torch.Tensor,
) -> list[torch.Tensor]:
    import torch_npu

    return [
        torch_npu.npu_quant_matmul(
            activation[index],
            weight[index],
            weight_scale[index],
            pertoken_scale=activation_scale[index],
            output_dtype=torch.bfloat16,
        )
        for index in range(activation.shape[0])
    ]


def channel_int8_bmm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    baseline_activation: torch.Tensor,
    baseline_weight: torch.Tensor,
) -> torch.Tensor:
    return flag_gems.bmm_w8a8(activation, weight, activation_scale, weight_scale)


class ChannelInt8BmmBenchmark(ParallelBenchmarkMixin, base.Benchmark):
    """Time BMM on prepared INT8 and BF16 inputs with the shared timer."""

    RECORD_PARALLEL_RESULTS = True
    DEFAULT_SHAPE_DESC = "B, M, N, K"
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            activation, weight, activation_scale, weight_scale = (
                make_channel_int8_bmm_inputs(shape, self.device, dtype)
            )
            # Both timed APIs receive prepared inputs; conversion stays outside timing.
            baseline_activation = (
                activation.float() * activation_scale[:, :, None]
            ).to(torch.bfloat16)
            baseline_weight = (
                weight.transpose(1, 2).float() * weight_scale[:, :, None]
            ).to(torch.bfloat16)
            yield (
                activation,
                weight,
                activation_scale,
                weight_scale,
                baseline_activation,
                baseline_weight.transpose(1, 2),
            )

    def should_forward_parallel_dtype(self, dtype_name):
        # Workers select their INT8 default without extending the shared CLI choices.
        return False


@pytest.mark.bmm_w8a8
@pytest.mark.parametrize(
    "baseline",
    ["bf16", "native_int8"] if "bmm_w8a8_int8" in vars(flag_gems) else ["bf16"],
)
@pytest.mark.skipif(
    "bmm_w8a8" not in vars(flag_gems),
    reason="backend does not export channel-wise INT8 BMM",
)
def test_perf_bmm_w8a8(baseline):
    torch_op = native_int8_bmm if baseline == "native_int8" else bf16_bmm
    bench = ChannelInt8BmmBenchmark(
        op_name="bmm_w8a8", torch_op=torch_op, dtypes=[torch.int8]
    )
    bench.set_gems(channel_int8_bmm)
    bench.run()


def capture(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        o = fn()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    return g, o


def measure(gs):
    values = [[] for g in gs]
    for i in range(31):
        for j in (range(len(gs)) if i % 2 else reversed(range(len(gs)))):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            a.record()
            for _ in range(5):
                gs[j].replay()
            b.record()
            b.synchronize()
            values[j].append(a.elapsed_time(b) * 200)
    return values


def main():
    if flag_gems.vendor_name == "metax" and not flag_gems.bmm_w8a8.__module__.endswith(
        "_metax.ops.bmm_w8a8"
    ):
        raise RuntimeError(
            "MetaX bmm_w8a8 was not registered; refusing to benchmark the generic fallback"
        )
    print(f"Implementation: {flag_gems.bmm_w8a8.__module__}", flush=True)
    torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    dtypes = (torch.int8,)
    if os.environ.get("BMM_EINSUM_SHAPES") == "1":
        # FlagOSTune f4f65a7d779a0df3b017e9a80b75a7b9d651c639, DeepSeek-V4-Flash.
        # Preserve source order and invocation counts; batch=1, N=1024, K=4096.
        m_counts = [
            (16384, 4310),
            (496, 1376),
            (512, 1376),
            (1, 688),
            (104, 688),
            (112, 688),
            (120, 688),
            (128, 688),
            (136, 688),
            (144, 688),
            (152, 688),
            (16, 688),
            (160, 688),
            (168, 688),
            (176, 688),
            (184, 688),
            (192, 688),
            (2, 688),
            (200, 688),
            (208, 688),
            (216, 688),
            (224, 688),
            (232, 688),
            (24, 688),
            (240, 688),
            (248, 688),
            (256, 688),
            (272, 688),
            (288, 688),
            (304, 688),
            (32, 688),
            (320, 688),
            (336, 688),
            (352, 688),
            (368, 688),
            (384, 688),
            (4, 688),
            (40, 688),
            (400, 688),
            (416, 688),
            (432, 688),
            (448, 688),
            (464, 688),
            (48, 688),
            (480, 688),
            (56, 688),
            (64, 688),
            (72, 688),
            (8, 688),
            (80, 688),
            (88, 688),
            (96, 688),
            (2048, 172),
            (4100, 172),
            (13021, 129),
            (1028, 43),
            (3805, 43),
            (3076, 43),
            (12991, 43),
            (13005, 43),
        ]
        shapes = [(1, m, 1024, 4096, count) for m, count in m_counts]
        shape_source = "fp8_einsum__int8_einsum"
    else:
        shapes = [
            (1, 16384, 7168, 1024, 12),
            (1, 16384, 2112, 7168, 5),
            (1, 448, 7168, 256, 4),
            (1, 14429, 7168, 1024, 4),
            (1, 14429, 2112, 7168, 1),
        ]
        shape_source = "bmm_w8a8_fp8__int8"
    for dt in dtypes:
        for bs, m, n, k, count in shapes:
            raw_a = torch.randn(bs, m, k, device="cuda", dtype=torch.bfloat16)
            raw_b = torch.randn(
                bs, n, k, device="cuda", dtype=torch.bfloat16
            ).transpose(1, 2)
            limit = 127
            sa = raw_a.float().abs().amax(2) / limit
            sb = raw_b.float().abs().amax(1) / limit
            aq = raw_a.float() / sa[:, :, None]
            bq = raw_b.float() / sb[:, None, :]
            aq = aq.round()
            bq = bq.round()
            a = aq.to(dt)
            b = bq.transpose(1, 2).contiguous().to(dt).transpose(1, 2)
            ref = (
                torch.bmm(a.float(), b.float()) * sa[:, :, None] * sb[:, None, :]
            ).bfloat16()
            f = lambda a=a, b=b, sa=sa, sb=sb: flag_gems.bmm_w8a8(a, b, sa, sb)
            base = lambda a=a, b=b, sa=sa, sb=sb: (
                torch.bmm(a.float(), b.float()) * sa[:, :, None] * sb[:, None, :]
            ).bfloat16()
            # Raw BF16 BMM is a separate lower-work baseline, not numerically equivalent.
            ab = raw_a
            bb = raw_b
            raw = lambda ab=ab, bb=bb: torch.bmm(ab, bb)
            out = f()
            torch.testing.assert_close(out, ref, atol=0.002, rtol=0.01)
            gs = []
            outs = []
            for fn in (f, base, raw):
                g, o = capture(fn)
                gs.append(g)
                outs.append(o)
            samples = measure(gs)
            us = [statistics.median(x) for x in samples]
            rows.append(
                dict(
                    dtype=str(dt),
                    implementation=flag_gems.bmm_w8a8.__module__,
                    shape=[bs, m, n, k],
                    batch_assumed=shape_source == "bmm_w8a8_fp8__int8",
                    shape_source=shape_source,
                    count=count,
                    candidate_us=us[0],
                    torch_channel_us=us[1],
                    raw_bf16_us=us[2],
                    speedup=us[1] / us[0],
                    raw_speedup=us[2] / us[0],
                    samples=samples,
                    correct=True,
                )
            )
            json.dump(
                rows, open(os.environ.get("BMM_RESULT", "results.json"), "w"), indent=2
            )
            print(
                rows[-1]["dtype"],
                [bs, m, n, k],
                us,
                "speedup",
                us[1] / us[0],
                "raw",
                us[2] / us[0],
                flush=True,
            )
            del gs, outs, g, o, out, ref, a, b, ab, bb, raw_a, raw_b, aq, bq
            gc.collect()
            torch.cuda.empty_cache()
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
