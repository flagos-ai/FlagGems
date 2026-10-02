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

# Metadata validation covers dense and stride-zero operands on the shared grid.

_CAPS = flag_gems.runtime.device

_DTYPES = list(consts.FLOAT_DTYPES) + [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
]
if _CAPS.support_fp8:
    _DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]
if _CAPS.support_fp64:
    _DTYPES += [torch.float64, torch.complex128]

_DTYPES += [torch.complex64]
_DTYPES = [
    dtype
    for dtype in _DTYPES
    if (dtype != torch.bfloat16 or _CAPS.support_bf16)
    and (dtype != torch.int64 or _CAPS.support_int64)
]

# Which optional checks a case performs. The expected values always come from the
# input's own metadata, so every case is a native-valid call; the empty subset omits
# all optional arguments.
_CHECK_SUBSETS = [
    [],
    ["size"],
    ["size", "stride"],
    ["size", "stride", "dtype"],
    ["size", "stride", "dtype", "device", "layout"],
]


def _case_fn(shape, dtype):
    del dtype
    for layout in ("contiguous", "expanded"):
        for checks in _CHECK_SUBSETS:
            yield base.BenchmarkCasePlan(
                shape={"input": shape},
                params={"checks": checks, "layout": layout},
                builder_args=(shape, layout),
            )


def _build_inputs_fn(plan, dtype, device):
    shape, layout = plan.builder_args
    inp = (
        torch.empty(shape, dtype=dtype, device=device)
        if layout == "contiguous"
        else torch.empty((), dtype=dtype, device=device).expand(shape)
    )
    metadata = {
        "size": list(inp.size()),
        "stride": list(inp.stride()),
        "dtype": inp.dtype,
        "device": inp.device,
        "layout": inp.layout,
    }
    # Flat positional arguments plus a trailing kwargs dict, which is the form
    # unpack_to_args_kwargs consumes.
    return inp, {name: metadata[name] for name in plan.params["checks"]}


class AssertTensorMetadataBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                tuple(shape)
                for shape in list(self.shapes)
                + [(), (256,), (1024, 1024), (20, 320, 15)]
            )
        )


@pytest.mark.assert_tensor_metadata
def test__assert_tensor_metadata():
    bench = AssertTensorMetadataBenchmark(
        op_name="_assert_tensor_metadata",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._assert_tensor_metadata,
        gems_op=getattr(flag_gems, "_assert_tensor_metadata", None),
        dtypes=_DTYPES,
    )
    bench.run()
