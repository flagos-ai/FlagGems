# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

# aten::requires_grad_ writes one autograd flag and returns its own input, so a
# timed run never depends on the element count. The shared default,
# comprehensive and caller-provided shape grids are kept; this operator's own
# shapes are appended to the framework extras.

REQUIRES_GRAD_SHAPES = [
    (256,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (64, 512, 512),
]

REQUIRES_GRAD_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

REQUIRES_GRAD_DTYPES = [
    dtype for dtype in REQUIRES_GRAD_DTYPES if _DTYPE_FLAGS.get(dtype, True)
]


def _case_fn(shape, dtype):
    # The clearing no-op is valid for every dtype; the setting and clearing
    # transitions plus the idempotent True repeat are only valid where the
    # dtype can carry a gradient.
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"initial_requires_grad": False, "requires_grad": False},
        builder_args=(shape, False, False),
    )
    if dtype.is_floating_point or dtype.is_complex:
        for initial, flag in ((False, True), (True, False), (True, True)):
            yield base.BenchmarkCasePlan(
                shape={"input": shape},
                params={"initial_requires_grad": initial, "requires_grad": flag},
                builder_args=(shape, initial, flag),
            )


def _build_inputs_fn(plan, dtype, device):
    # No element is read, so the payload stays uninitialized. fresh_inputs
    # restores this exact initial flag before every warmup and measured sample,
    # so the clearing case really starts from True.
    shape, initial, flag = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    if initial:
        torch.ops.aten.requires_grad_(inp, True)
    return inp, {"requires_grad": flag}


class RequiresGradBenchmark(OperatorBenchmark):
    def set_more_shapes(self):
        # Caller shape files still take precedence: set_shapes resolves them
        # before these extras are appended.
        return super().set_more_shapes() + REQUIRES_GRAD_SHAPES


@pytest.mark.requires_grad_
def test_requires_grad_():
    bench = RequiresGradBenchmark(
        op_name="requires_grad_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.requires_grad_,
        gems_op=getattr(flag_gems, "requires_grad_", None),
        dtypes=REQUIRES_GRAD_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
