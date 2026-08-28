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


def _input_fn(shape, dtype, device):
    if dtype.is_complex:
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.randn(shape, dtype=float_dtype, device=device)
        imag = torch.randn(shape, dtype=float_dtype, device=device)
        input_tensor = torch.complex(real, imag).to(dtype)
    elif dtype.is_floating_point:
        input_tensor = torch.randn(shape, dtype=dtype, device=device)
    else:
        input_tensor = torch.randn(shape, device=device).to(dtype)
    yield (input_tensor,)


class ConjPhysicalInplaceBenchmark(base.GenericBenchmarkExcluse3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        conj_physical_shapes = [
            (256,),
            (2048, 2048),
            (128, 512, 256),
            (32, 64),
            (512, 1024),
            (2, 3, 4),
        ]
        self.shapes = conj_physical_shapes

    def set_more_shapes(self):
        return None


@pytest.mark.conj_physical_
def test_conj_physical_():
    if flag_gems.vendor_name == "ascend":
        # Ascend NPU: kernel-mode event timing is unstable, use operator mode.
        # Native conj_physical_ has no complex kernel on NPU, so complex
        # dtypes are excluded from the baseline comparison.
        from .conftest import Config

        Config.mode = consts.BenchMode.OPERATOR
        dtypes = consts.FLOAT_DTYPES + consts.INT_DTYPES
    else:
        dtypes = consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.COMPLEX_DTYPES

    bench = ConjPhysicalInplaceBenchmark(
        input_fn=_input_fn,
        op_name="conj_physical_",
        torch_op=torch.conj_physical_,
        dtypes=dtypes,
    )

    bench.set_gems(flag_gems.conj_physical_)
    bench.run()
