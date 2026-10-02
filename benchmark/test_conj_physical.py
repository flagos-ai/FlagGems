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

# torch.randn has no FP8 implementation, so FP8 inputs are built from an FP32
# tensor; int8/uint8 go through the existing truncating path.
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
INT8_DTYPES = [torch.int8, torch.uint8]


def _supported_fp8_dtypes(device):
    """Keep only the FP8 dtypes the current device can actually materialize."""
    supported = []
    for dtype in FP8_DTYPES:
        try:
            torch.randn(1, device=device).to(dtype)
        except Exception:
            continue
        supported.append(dtype)
    return supported


def _input_fn(shape, dtype, device):
    if dtype.is_complex:
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.randn(shape, dtype=float_dtype, device=device)
        imag = torch.randn(shape, dtype=float_dtype, device=device)
        input_tensor = torch.complex(real, imag).to(dtype)
    elif dtype in FP8_DTYPES:
        input_tensor = torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    elif dtype.is_floating_point:
        input_tensor = torch.randn(shape, dtype=dtype, device=device)
    else:
        input_tensor = torch.randn(shape, device=device).to(dtype)
    yield (input_tensor,)


class Conj_physicalBenchmark(base.GenericBenchmarkExcluse3D):
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


@pytest.mark.conj_physical
def test_conj_physical():
    if "npu" in flag_gems.device or "ascend" in flag_gems.device.lower():
        # Ascend NPU: kernel mode event timing is unstable, use operator mode
        from .conftest import Config

        Config.mode = consts.BenchMode.OPERATOR
        dtypes = consts.FLOAT_DTYPES + consts.INT_DTYPES + INT8_DTYPES
    else:
        dtypes = (
            consts.FLOAT_DTYPES
            + consts.INT_DTYPES
            + INT8_DTYPES
            + _supported_fp8_dtypes(flag_gems.device)
            + consts.COMPLEX_DTYPES
        )

    bench = Conj_physicalBenchmark(
        input_fn=_input_fn,
        op_name="conj_physical",
        torch_op=torch.conj_physical,
        dtypes=dtypes,
    )

    bench.set_gems(flag_gems.conj_physical)
    bench.run()


@pytest.mark.underscore_conj_physical
def test__conj_physical():
    # The private op only differs from the public one for real dtypes (where it
    # still allocates), so benchmark the complex path it actually accelerates.
    bench = Conj_physicalBenchmark(
        input_fn=_input_fn,
        op_name="_conj_physical",
        torch_op=torch.ops.aten._conj_physical,
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.set_gems(flag_gems.ops._conj_physical)
    bench.run()


def _out_input_fn(shape, dtype, device):
    """Same inputs as ``_input_fn``, plus the pre-allocated ``out`` target."""
    for args in _input_fn(shape, dtype, device):
        yield args[0], {"out": torch.empty_like(args[0])}


@pytest.mark.underscore_conj_physical_out
def test__conj_physical_out():
    bench = Conj_physicalBenchmark(
        input_fn=_out_input_fn,
        op_name="_conj_physical_out",
        torch_op=torch.ops.aten._conj_physical.out,
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.set_gems(flag_gems.ops._conj_physical_out)
    bench.run()
