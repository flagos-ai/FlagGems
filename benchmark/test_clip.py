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

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, -0.5, 0.5
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, None, 0.5
        yield inp, -0.5, None


@pytest.mark.clip
def test_clip():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="clip",
        torch_op=torch.clip,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.clip_
def test_clip_inplace():
    bench = base.GenericBenchmark(
        input_fn=_input_fn,
        op_name="clip_",
        torch_op=torch.clip_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
