# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flag_gems

from . import base, consts


def reference(x):
    gate, up = x.chunk(2, dim=-1)
    gate, up = gate.clamp(max=7.0), up.clamp(-7.0, 7.0)
    return (gate * torch.sigmoid(1.702 * gate)) * (up + 1.0)


class OperatorBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 384),
            (4, 768),
            (64, 768),
            (4096, 384),
            (5089, 768),
            (8192, 1536),
        ]
        self.shape_desc = "synthetic operator dimensions"

    def get_input_iter(self, dtype):
        for m, i in self.shapes:
            yield (torch.randn((m, 2 * i), device=self.device, dtype=dtype),)


@pytest.mark.swiglu_oai
def test_perf_swiglu_oai():
    bench = OperatorBenchmark(
        op_name="swiglu_oai",
        torch_op=reference,
        gems_op=flag_gems.swiglu_oai,
        dtypes=[torch.bfloat16, torch.float16],
    )
    bench.run()
