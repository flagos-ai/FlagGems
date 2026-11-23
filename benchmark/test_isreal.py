# Copyright 2026, The FlagOS Contributors.
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


def _input_fn(shape, dtype, device):
    if dtype.is_complex:
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.randn(shape, dtype=float_dtype, device=device)
        imag = torch.randn(shape, dtype=float_dtype, device=device)
        # Force a mixed result so both the True and the False branch of the
        # predicate are exercised (torch.randn never yields an exact zero).
        imag[::2] = 0
        yield (torch.complex(real, imag).to(dtype),)
    else:
        yield (torch.randn(shape, dtype=dtype, device=device),)


@pytest.mark.isreal
def test_isreal():
    bench = base.UnaryPointwiseBenchmark(
        input_fn=_input_fn,
        op_name="isreal",
        torch_op=torch.isreal,
        dtypes=consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES,
    )
    bench.set_gems(flag_gems.isreal)
    bench.run()
