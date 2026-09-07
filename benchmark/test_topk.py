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

from . import base, consts
from .conftest import Config


class TopKBenchmark(base.GenericBenchmark2DOnly):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64, 64),
            (4096, 4096),
            (10000, 256),
            (10000, 65536),
            (4, 128),
            (8, 256),
            (64, 128, 8),
            (64, 1024, 32),
            (64, 8192, 128),
            (128, 32768, 256),
            ((4, 128, 64), 5),
            ((4, 128, 64), 64),
            ((8, 512, 32), 32),
            ((16, 1024, 256), 256),
        ]


def _input_fn(shape, dtype, device):
    if len(shape) == 2 and isinstance(shape[0], (tuple, list)):
        x_shape, k = shape
        x = torch.randn(x_shape, device=device, dtype=dtype)
        yield {"x": x, "k": k, "dim": -1},
    elif len(shape) == 3:
        m, n, k = shape
        x = torch.randn((m, n), device=device, dtype=dtype)
        yield {"x": x, "k": k, "dim": -1},
    else:
        x = torch.randn(shape, device=device, dtype=dtype)
        k = 5 if shape[-1] > 5 else shape[-1]
        yield {"x": x, "k": k, "dim": -1},
    # TODO:  Currently only support sorted == True and only support topk in last dimension
    # if Config.bench_level == BenchLevel.COMPREHENSIVE:
    #     k = 5 if shape[0] > 5 else shape[0]
    #     yield {"x": x, "k": k, "dim": 0},
    #     yield {"x": x, "k": k, "dim": -1, "sorted": False},


@pytest.mark.topk
def test_topk():
    bench = TopKBenchmark(
        op_name="topk",
        input_fn=_input_fn,
        torch_op=torch.topk,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


class _TopKGraphMode(Enum):
    NPUGRAPH = "npugraph"


def _topk_fp8_reference(x, q, scale, k):
    return torch.topk(x, k)


def _topk_fp8_impl(x, q, scale, k):
    return flag_gems.topk_w8a16_fp8(q, scale, k, group_size=x.shape[-1])


class TopKFp8Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (4, 128, 8),
            (8, 256, 16),
            (64, 1024, 32),
            (64, 4096, 64),
            (64, 8192, 128),
            (128, 32768, 256),
        ]

    def get_input_iter(self, dtype):
        for m, n, k in self.shapes:
            torch.manual_seed(42)
            x = torch.randn((m, n), dtype=dtype)
            scale = (
                (x.float().abs().amax(-1, keepdim=True) / 448).clamp_min(1e-8).to(dtype)
            )
            q = (x.float() / scale.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
            reference = q.float() * scale.float()
            x, q, scale = x.to(self.device), q.to(self.device), scale.to(self.device)
            v, i = _topk_fp8_impl(x, q, scale, k)
            torch.testing.assert_close(
                v.cpu(), torch.topk(reference, k).values.bfloat16(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                v.cpu(), torch.gather(reference, -1, i.cpu()).bfloat16(), rtol=0, atol=0
            )
            yield x, q, scale, k

    def record_shapes(self, x, q, scale, k):
        return (*x.shape, k)

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


@pytest.mark.topk_w8a16_fp8
def test_topk_w8a16_fp8_npugraph(monkeypatch):
    if flag_gems.device != "npu":
        pytest.skip("Ascend only")
    # This dedicated test always uses actual NPU Graph. Keep its report label
    # local so other benchmarks and their global CLI modes are unchanged.
    monkeypatch.setattr(Config, "mode", _TopKGraphMode.NPUGRAPH)
    bench = TopKFp8Benchmark(
        op_name="topk_w8a16_fp8", torch_op=_topk_fp8_reference, dtypes=[torch.bfloat16]
    )
    bench.set_gems(_topk_fp8_impl)
    bench.run()
