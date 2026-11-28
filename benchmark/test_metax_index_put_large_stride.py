# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Compare unchanged index_put throughput on small and >int32 paged strides."""

import pytest
import torch

import flag_gems

from . import base

pytestmark = [
    pytest.mark.index_put_,
    pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX address arithmetic"
    ),
]


class PagedIndexPutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(128, 128), (22002, 98304)]

    def get_input_iter(self, dtype):
        for pages, width in self.shapes:
            stride = 102400 if width == 98304 else width
            x = torch.empty_strided(
                (pages, width), (stride, 1), device=self.device, dtype=dtype
            )
            indices = torch.tensor(
                [0, pages // 2, pages - 2, pages - 1],
                device=self.device,
                dtype=torch.int64,
            )
            values = torch.ones(4, width, device=self.device, dtype=dtype)
            yield x, [indices], values, False


def test_metax_paged_index_put_benchmark():
    bench = PagedIndexPutBenchmark(
        op_name="index_put_", torch_op=torch.index_put_, dtypes=[torch.float32]
    )
    bench.set_gems(flag_gems.index_put_)
    bench.run()
