# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flag_gems
from flag_gems.fused.bf16_small_m_linear import bf16_small_m_linear

from . import base


class SmallMLinearBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 1024, 6144), (4, 1024, 6144), (16, 1024, 6144)]

    def get_input_iter(self, dtype):
        for m, n, k in self.shapes:
            yield (
                torch.randn((m, k), dtype=dtype, device=self.device),
                torch.randn((n, k), dtype=dtype, device=self.device),
            )


def _small_m(input, weight):
    return bf16_small_m_linear(input, weight, device_sm=90)


@pytest.mark.bf16_small_m_linear
def test_bf16_small_m_linear():
    if flag_gems.vendor_name != "nvidia" or not torch.cuda.is_available():
        pytest.skip("SM90 CUDA specialization")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 CUDA specialization")
    bench = SmallMLinearBenchmark(
        op_name="bf16_small_m_linear",
        torch_op=torch.nn.functional.linear,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_small_m)
    bench.run()
