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

import flag_gems

from . import base, consts

# bfloat16 has no kernel on every backend; this is a static capability query, so
# case listing stays free of device work and tensor allocations.
_FLOAT_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]

# The framework families build their inputs with utils.generate_tensor_input,
# which only knows the float, int16/int32 and bool groups. The remaining
# supported groups (int8, uint8, int64, fp8, float64, complex64) are exercised by
# the torch.empty-based layout benchmark below.
_FAMILY_DTYPES = _FLOAT_DTYPES + list(consts.INT_DTYPES) + list(consts.BOOL_DTYPES)
_LAYOUT_DTYPES = (
    _FLOAT_DTYPES
    + list(consts.INT_DTYPES)
    + list(consts.BOOL_DTYPES)
    + [torch.int8, torch.uint8, torch.complex64]
    + ([torch.int64] if flag_gems.runtime.device.support_int64 else [])
    + ([torch.float64] if flag_gems.runtime.device.support_fp64 else [])
    + (
        [torch.float8_e4m3fn, torch.float8_e5m2]
        if flag_gems.runtime.device.support_fp8
        else []
    )
)

# Layout scales this family adds on top of the shared core/comprehensive shapes;
# the layout rows below vary the stride pattern, not the scale.
_LAYOUT_EXTRA_SHAPES = ((64, 512, 512),)
_LAYOUT_ROWS = ("dense", "transposed", "offset")


def _layout_input(shape, layout, dtype, device):
    """Materialise one layout row from unread memory.

    The copy is the only consumer of its payload, and the benchmark preflight
    executes the candidate without comparing values, so uninitialised memory is
    valid here and keeps the builder independent of
    ``utils.generate_tensor_input``, which has no branch for fp8.
    """
    if layout == "dense":
        return torch.empty(shape, dtype=dtype, device=device)
    if layout == "offset":
        flat = torch.empty((math.prod(shape) + 1,), dtype=dtype, device=device)
        return flat[1:].view(shape)
    store = torch.empty(
        tuple(shape[:-2]) + (shape[-1], shape[-2]), dtype=dtype, device=device
    )
    return store.transpose(-1, -2)


def _case_fn(shape, dtype):
    del dtype
    for layout in _LAYOUT_ROWS:
        if layout == "transposed" and len(shape) < 2:
            continue
        yield base.BenchmarkCasePlan(
            shape={"input": tuple(shape), "input_layout": layout},
            params={"input_layout": layout},
            builder_args=(tuple(shape), layout),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, layout = plan.builder_args
    return _layout_input(shape, layout, dtype, device), {}


class CopyLayoutBenchmark(base.GenericBenchmark):
    """``_test_warn_in_autograd`` over non-contiguous input layouts."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Keep the shared core/comprehensive shapes (or the caller's explicit
        # --shape_file selection) and add the layout scales on top; the dedup
        # preserves the framework ordering.
        self.shapes = list(
            dict.fromkeys(
                [tuple(shape) for shape in self.shapes] + list(_LAYOUT_EXTRA_SHAPES)
            )
        )


# All three tests benchmark the single public operator `_test_warn_in_autograd`:
# the candidate is resolved through that one name (the out form passes `out=` to
# the same callable), so `op_name` must stay the operator name for the override
# lookup and the case-list identity to resolve.
@pytest.mark.test_warn_in_autograd
def test__test_warn_in_autograd():
    bench = base.UnaryPointwiseBenchmark(
        op_name="_test_warn_in_autograd",
        torch_op=torch.ops.aten._test_warn_in_autograd,
        gems_op=getattr(flag_gems, "_test_warn_in_autograd", None),
        dtypes=_FAMILY_DTYPES,
    )
    bench.run()


@pytest.mark.test_warn_in_autograd
def test__test_warn_in_autograd_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="_test_warn_in_autograd",
        torch_op=torch.ops.aten._test_warn_in_autograd,
        gems_op=getattr(flag_gems, "_test_warn_in_autograd", None),
        dtypes=_FAMILY_DTYPES,
    )
    bench.run()


@pytest.mark.test_warn_in_autograd
def test__test_warn_in_autograd_layout():
    bench = CopyLayoutBenchmark(
        op_name="_test_warn_in_autograd",
        torch_op=torch.ops.aten._test_warn_in_autograd,
        gems_op=getattr(flag_gems, "_test_warn_in_autograd", None),
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=_LAYOUT_DTYPES,
    )
    bench.run()
