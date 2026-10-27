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

from flag_gems.utils import shape_utils

from . import base, consts


class UnbindCopyBenchmark(base.GenericBenchmark):
    """Benchmark for unbind_copy operator.
    Overrides set_shapes to use shapes suitable for unbind operations."""

    def set_shapes(self, shape_file_path=None):
        UNBIND_COPY_SHAPES = (
            (2, 3),
            (4, 8),
            (16, 32),
            (4, 8, 16),
            (32, 64, 128),
            (2, 4, 8, 16),
        )
        self.shapes = UNBIND_COPY_SHAPES

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, bench_fn_args, latency):
        inp = bench_fn_args[0]
        io_amount = shape_utils.size_in_bytes(inp) * 2
        return io_amount * 1e-9 / (latency * 1e-3)


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    dim = 0
    yield inp, dim


@pytest.mark.unbind_copy
def test_unbind_copy():
    bench = UnbindCopyBenchmark(
        op_name="unbind_copy",
        torch_op=torch.unbind_copy,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
