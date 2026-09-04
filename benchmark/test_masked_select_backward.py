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

import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base, consts, utils


class MaskedSelectBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1,),
            (1024,),
            (4096,),
            (4097,),
            (128, 128),
            (1024, 1024),
            (4096, 4096),
        ]
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += [(17, 17), (8192,), (256, 256), (8192, 4096)]

    def set_more_metrics(self):
        return ["gbps"]


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    mask = torch.rand(shape, device=device) < 0.5
    grad = utils.generate_tensor_input((int(mask.sum()),), dtype, device)
    yield grad, inp, mask


def _get_gbps(bench_fn_args, latency):
    grad, inp, mask = bench_fn_args
    io_amount = sum(shape_utils.size_in_bytes(item) for item in (grad, inp, mask, inp))
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.masked_select_backward
def test_masked_select_backward():
    bench = MaskedSelectBackwardBenchmark(
        op_name="masked_select_backward",
        torch_op=torch.ops.aten.masked_select_backward.default,
        gems_op=flag_gems.masked_select_backward,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )
    bench.run()
