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

from . import base, consts, utils

MODE_DTYPES = [
    pytest.param(
        dtype,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "mthreads"
            and dtype in (torch.float16, torch.bfloat16, torch.int16),
            reason=(
                "MThreads native torch.mode raises 'MUSA error: misaligned address' "
                "during repeated benchmarking for FP16/BF16/INT16 at shapes "
                "(64, 64), (256, 256), and (1024, 1024); skip these dtypes "
                "pending a native backend fix."
            ),
        ),
    )
    for dtype in consts.INT_DTYPES + consts.FLOAT_DTYPES
]


class ModeBenchmark(base.GenericBenchmark2DOnly):
    def set_dtypes(self, user_desired_dtypes):
        dtype = self.dtypes[0]
        if user_desired_dtypes and dtype not in user_desired_dtypes:
            pytest.skip(f"{dtype} was not selected by --dtypes")
        self.to_bench_dtypes = [dtype]

    def set_more_shapes(self):
        return [(1024, 1), (1024, 512), (16, 128 * 1024), (8, 256 * 1024)]


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": -1},


@pytest.mark.mode
@pytest.mark.parametrize("dtype", MODE_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_perf_mode(dtype):
    bench = ModeBenchmark(
        input_fn=_input_fn,
        op_name="mode",
        torch_op=torch.mode,
        dtypes=[dtype],
    )
    bench.run()
