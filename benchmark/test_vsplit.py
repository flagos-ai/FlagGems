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

from . import base, consts


def vsplit_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)
    # Use integer split (equal chunks)
    sections = 2
    yield inp, sections
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, 4


def vsplit_array_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)
    # Use list-of-indices split
    indices = [shape[0] // 3, 2 * shape[0] // 3]
    yield inp, indices


class VsplitBenchmark(base.GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes = [
            (64, 128),
            (128, 256),
            (256, 512),
            (512, 1024),
        ]

        for shape in shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.vsplit
def test_perf_vsplit():
    def vsplit_wrapper(input, sections):
        return torch.ops.aten.vsplit.int(input, sections)

    bench = VsplitBenchmark(
        input_fn=vsplit_input_fn,
        op_name="vsplit",
        torch_op=vsplit_wrapper,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.vsplit
def test_perf_vsplit_array():
    def vsplit_array_wrapper(input, indices):
        return torch.ops.aten.vsplit.array(input, indices)

    bench = VsplitBenchmark(
        input_fn=vsplit_array_input_fn,
        op_name="vsplit",
        torch_op=vsplit_array_wrapper,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
