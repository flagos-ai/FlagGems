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

from typing import Generator

import pytest
import torch

from flag_gems.utils import shape_utils

from . import base, consts


class NestedTensorFromMaskLeftAlignedBenchmark(base.Benchmark):
    """
    Benchmark class for _nested_tensor_from_mask_left_aligned.
    The operation only reads the 2D boolean padding mask (the 3D tensor `t` is
    used solely for shape validation), returning a bool.
    """

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        mask = args[1]
        io_amount = shape_utils.size_in_bytes(mask)
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, dtype) -> Generator:
        shapes = [(8, 128), (64, 256), (512, 512), (2048, 1024)]
        for N, L in shapes:
            D = 64
            t = torch.randn(N, L, D, dtype=dtype, device=self.device)
            lengths = torch.randint(0, L + 1, (N,), device=self.device)
            mask = torch.arange(L, device=self.device)[None, :] < lengths[:, None]
            yield t, mask


@pytest.mark.nested_tensor_from_mask_left_aligned
def test_nested_tensor_from_mask_left_aligned():
    bench = NestedTensorFromMaskLeftAlignedBenchmark(
        op_name="nested_tensor_from_mask_left_aligned",
        torch_op=torch._nested_tensor_from_mask_left_aligned,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
