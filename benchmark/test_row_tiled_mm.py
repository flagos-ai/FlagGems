# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flag_gems

from . import base


def native_row_tiles(a, b):
    out = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
    for start in range(0, a.shape[0], 64):
        flag_gems.mm_out(a[start : start + 64], b, out=out[start : start + 64])
    return out


class RowTiledMMBenchmark(base.Benchmark):
    def get_input_iter(self, dtype):
        for m, n, k in self.shapes:
            yield torch.randn((m, k), device=self.device, dtype=dtype), torch.randn(
                (n, k), device=self.device, dtype=dtype
            ).t()


@pytest.mark.row_tiled_mm
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX row-tiled MM")
def test_row_tiled_mm_benchmark():
    bench = RowTiledMMBenchmark(
        op_name="row_tiled_mm", torch_op=native_row_tiles, dtypes=[torch.bfloat16]
    )
    bench.set_gems(flag_gems.row_tiled_mm)
    bench.run()
