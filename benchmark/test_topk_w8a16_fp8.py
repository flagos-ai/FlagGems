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
from enum import Enum

import pytest
import torch

import flag_gems

from . import base
from .conftest import Config

GROUP_SIZE = 128
FP8_DTYPE = (
    torch.float8_e4m3fn
    if flag_gems.vendor_name in ("ascend", "hygon", "mthreads", "nvidia")
    else torch.float8_e5m2
)


def _fp8_available():
    if flag_gems.device == "npu":
        return torch.npu.is_available() and hasattr(torch, "float8_e4m3fn")
    if flag_gems.device == "musa":
        return torch.musa.is_available() and hasattr(torch, "float8_e4m3fn")
    return (
        torch.cuda.is_available()
        and hasattr(torch, "float8_e5m2")
        and (
            flag_gems.vendor_name != "nvidia"
            or torch.cuda.get_device_capability()[0] >= 9
        )
    )


def _quantize_fp8_grouped(x, group_size=GROUP_SIZE):
    fp8_info = torch.finfo(FP8_DTYPE)
    *leading, n = x.shape
    padded = (n + group_size - 1) // group_size * group_size
    x_pad = torch.nn.functional.pad(x.float(), (0, padded - n))
    grouped = x_pad.reshape(*leading, padded // group_size, group_size)
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(min=1e-8)
    q = (grouped / scale).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)
    return (
        q.reshape(*leading, padded)[..., :n].contiguous(),
        scale.squeeze(-1).to(x.dtype).contiguous(),
    )


def _dequant_fp8(x_fp8, x_scale, group_size=GROUP_SIZE):
    *leading, n = x_fp8.shape
    num_groups = x_scale.shape[-1]
    padded = num_groups * group_size
    x_pad = torch.nn.functional.pad(x_fp8.float(), (0, padded - n))
    grouped = x_pad.reshape(*leading, num_groups, group_size)
    dequant = grouped * x_scale.unsqueeze(-1).float()
    return dequant.reshape(*leading, padded)[..., :n].to(x_scale.dtype)


def _torch_topk_w8a16(x_fp8, x_scale, k, dequant, group_size):
    return torch.topk(dequant, k, dim=-1, largest=True, sorted=True)


def _gems_bf16_topk(x_fp8, x_scale, k, dequant, group_size):
    return flag_gems.topk(dequant, k, dim=-1, largest=True, sorted=True)


def _gems_topk_w8a16(x_fp8, x_scale, k, dequant, group_size):
    return flag_gems.topk_w8a16_fp8(
        x_fp8, x_scale, k, dim=-1, largest=True, sorted=True, group_size=group_size
    )


class TopKFp8W8A16Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        if flag_gems.vendor_name in ("ascend", "nvidia"):
            self.shapes = [
                (4, 128, 8),
                (8, 256, 16),
                (64, 1024, 32),
                (64, 4096, 64),
                (64, 8192, 128),
                (128, 32768, 256),
            ]
            return
        self.shapes = [
            (64, 128, 8),
            (256, 256, 8),
            (128, 1024, 16),
            (64, 4096, 32),
            (32, 8192, 64),
            (16, 16384, 128),
            (8, 32768, 256),
        ]

    def get_input_iter(self, dtype):
        for m, n, k in self.shapes:
            torch.manual_seed(5966)
            if flag_gems.device == "npu":
                # Ascend quantizes on CPU because the device need not support FP8 casts.
                x = torch.randn((m, n), dtype=dtype)
                group_size = n
                x_fp8, x_scale = _quantize_fp8_grouped(x, group_size=group_size)
                reference = x_fp8.float() * x_scale.float()
                x_fp8 = x_fp8.view(torch.uint8).to(self.device).view(FP8_DTYPE)
                x_scale = x_scale.to(self.device)
                values, indices = flag_gems.topk_w8a16_fp8(
                    x_fp8, x_scale, k, group_size=group_size
                )
                torch.testing.assert_close(
                    values.cpu(),
                    torch.topk(reference, k).values.to(dtype),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    values.cpu(),
                    torch.gather(reference, -1, indices.cpu()).to(dtype),
                    rtol=0,
                    atol=0,
                )
                dequant = x.to(self.device)
            else:
                x = torch.randn((m, n), dtype=dtype, device=self.device)
                group_size = n if flag_gems.vendor_name == "nvidia" else GROUP_SIZE
                x_fp8, x_scale = _quantize_fp8_grouped(x, group_size=group_size)
                dequant = (
                    x
                    if flag_gems.vendor_name == "nvidia"
                    else _dequant_fp8(x_fp8, x_scale, group_size=group_size)
                )
            yield x_fp8, x_scale, k, dequant, group_size


class AscendTopKFp8W8A16Benchmark(TopKFp8W8A16Benchmark):
    def get_latency(self, op, *args, **kwargs):
        fn = lambda: op(*args, **kwargs)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(5):
                fn()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            for _ in range(100):
                fn()
        for _ in range(10):
            graph.replay()
        torch.npu.synchronize()
        starts = [torch.npu.Event(enable_timing=True) for _ in range(30)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(30)]
        for start, end in zip(starts, ends):
            start.record()
            graph.replay()
            end.record()
        torch.npu.synchronize()
        return (
            statistics.median(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
            / 100
        )


class AscendGraphMode(Enum):
    NPUGRAPH = "npugraph"


@pytest.mark.topk_w8a16_fp8
@pytest.mark.skipif(
    flag_gems.vendor_name
    not in ("ascend", "thead", "hygon", "mthreads", "nvidia", "metax"),
    reason="topk_w8a16_fp8 requires an implemented backend",
)
@pytest.mark.skipif(not _fp8_available(), reason="required FP8 format is unavailable")
@pytest.mark.parametrize(
    "baseline",
    (
        ["torch"]
        if flag_gems.vendor_name in ("ascend", "nvidia")
        else ["torch", "flaggems"]
    ),
)
def test_topk_w8a16_fp8(baseline, monkeypatch):
    if flag_gems.device == "npu":
        monkeypatch.setattr(Config, "mode", AscendGraphMode.NPUGRAPH)
        benchmark_class = AscendTopKFp8W8A16Benchmark
    else:
        benchmark_class = TopKFp8W8A16Benchmark
    bench = benchmark_class(
        op_name="topk_w8a16_fp8",
        torch_op=_torch_topk_w8a16 if baseline == "torch" else _gems_bf16_topk,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_topk_w8a16)
    print(
        f"FP8 format: {FP8_DTYPE}; BF16 baseline: {baseline}; input quantization excluded"
    )
    bench.run()
