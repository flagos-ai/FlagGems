# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flag_gems

from . import base


class DynQuantMatmul4BitBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1, 64, 64), (7, 128, 96), (32, 1024, 1024), (32, 4096, 4096)]
        self.shape_desc = "M, N, K"

    def get_input_iter(self, cur_dtype):
        for m, n, k in self.shapes:
            for group in [k, 32] if cur_dtype == torch.float32 and k != 32 else [k]:
                x = torch.randn((m, k), device=self.device, dtype=cur_dtype)
                w = torch.randint(
                    0, 256, (n, k // 2), device=self.device, dtype=torch.uint8
                )
                scales = torch.rand((n, k // group), device=self.device)
                bias = torch.randn(n, device=self.device)
                packed = flag_gems._dyn_quant_pack_4bit_weight(
                    w, scales, bias, group, k, n
                )
                yield x, packed, group, k, n


@pytest.mark.dyn_quant_matmul_4bit
@pytest.mark.skipif(
    flag_gems.vendor_name
    in ("nvidia", "hygon", "iluvatar", "ascend", "mthreads", "metax"),
    reason=(
        "aten::_dyn_quant_matmul_4bit has no native device kernel on this backend; "
        "CPU fallback is not a performance baseline"
    ),
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dyn_quant_matmul_4bit(dtype):
    bench = DynQuantMatmul4BitBenchmark(
        op_name="dyn_quant_matmul_4bit",
        torch_op=torch.ops.aten._dyn_quant_matmul_4bit.default,
        gems_op=flag_gems._dyn_quant_matmul_4bit,
        dtypes=[dtype],
    )
    bench.run()
