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

from . import base, consts


def bincount_input_fn(shape, dtype, device):
    if shape[0] > 1_000_000:
        return

    n = shape[0]
    for num_classes in [10, 256, 4096]:
        inp = torch.randint(0, num_classes, (n,), dtype=torch.int64, device=device)

        yield inp, {}

        yield inp, {"minlength": max(512, num_classes * 2)}


@pytest.mark.bincount
def test_bincount():
    bench = base.GenericBenchmark(
        input_fn=bincount_input_fn,
        op_name="bincount",
        torch_op=torch.bincount,
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.bincount)
    bench.run()


def bincount_weighted_input_fn(shape, dtype, device):
    if shape[0] > 1_000_000:
        return

    n = shape[0]
    for num_classes in [10, 256, 4096]:
        inp = torch.randint(0, num_classes, (n,), dtype=torch.int64, device=device)
        weights = torch.randn((n,), dtype=dtype, device=device)

        yield inp, {"weights": weights}

        yield inp, {"weights": weights, "minlength": max(512, num_classes * 2)}


@pytest.mark.bincount
def test_bincount_weighted():
    bench = base.GenericBenchmark(
        input_fn=bincount_weighted_input_fn,
        op_name="bincount_weighted",
        torch_op=torch.bincount,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.bincount)
    bench.run()
