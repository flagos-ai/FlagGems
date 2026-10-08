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


def dsplit_input_fn(shape, dtype, device):
    inp = base.generate_tensor_input(shape, dtype, device)
    # Use integer split (equal chunks)
    sections = 2
    yield inp, sections
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, 4


class DsplitBenchmark(base.GenericBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        shapes = [
            (64, 128, 32),
            (128, 256, 64),
            (256, 512, 128),
            (512, 1024, 256),
        ]

        for shape in shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.dsplit
def test_perf_dsplit():
    def dsplit_wrapper(input, sections):
        return torch.ops.aten.dsplit.int(input, sections)

    bench = DsplitBenchmark(
        input_fn=dsplit_input_fn,
        op_name="dsplit",
        torch_op=dsplit_wrapper,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
