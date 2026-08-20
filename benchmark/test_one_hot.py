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

import math

import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    if not isinstance(shape, tuple):
        return

    numel = math.prod(shape)
    if numel == 0:
        return

    if base.Config.bench_level != consts.BenchLevel.COMPREHENSIVE:
        num_classes_list = [16, 64]
    else:
        num_classes_list = [16, 64, 256]

    max_output_elems = 100_000_000

    for num_classes in num_classes_list:
        if numel * num_classes > max_output_elems:
            continue
        inp = torch.randint(0, num_classes, shape, device=device, dtype=torch.int64)
        inp.view(-1)[0] = num_classes - 1
        yield inp, num_classes
        yield inp, -1


@pytest.mark.one_hot
def test_one_hot():
    bench = base.GenericBenchmark(
        op_name="one_hot",
        input_fn=_input_fn,
        torch_op=torch.nn.functional.one_hot,
        dtypes=[torch.int64],
    )
    bench.run()
