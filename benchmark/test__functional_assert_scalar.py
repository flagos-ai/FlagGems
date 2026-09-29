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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

MSG = "functional_assert_scalar benchmark"

# The operator copies dep_token, so a case measures the copy volume of the token. The
# token stays contiguous, so these scales measure size rather than stride handling.
DEFAULT_SCALES = [
    (2**20,),
    (2**24,),
    (4096, 4096),
    (64, 512, 512),
    (16, 1024, 1024, 16),
]

# Static device capability gating, so listing and execution see the same dtype set.
BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES + [torch.int32]
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _valid_dep_token_shape(shape):
    """True for extents a torch allocation accepts: exact non-negative ints."""
    if not isinstance(shape, (list, tuple)):
        return False
    return all(
        isinstance(extent, int) and not isinstance(extent, bool) and extent >= 0
        for extent in shape
    )


def _check_dep_token_shape(shape):
    # Consumed as an error, never as a filter: a requested row is either valid or the
    # run must fail before any input tensor is allocated.
    if not _valid_dep_token_shape(shape):
        raise ValueError(
            f"dep_token shape {shape!r} is invalid: every extent must be a non-negative "
            "int (use () for a 0-D token)"
        )
    return tuple(shape)


def _case_fn(shape, dtype):
    del dtype
    dep_token_shape = _check_dep_token_shape(shape)
    yield base.BenchmarkCasePlan(
        shape={"dep_token": list(dep_token_shape)},
        params={"self": 1.0, "assert_msg": MSG},
        builder_args=(dep_token_shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    # Recheck the plan before allocating: a case built outside the shape loader must not
    # reach a confusing allocation error.
    dep_token_shape = _check_dep_token_shape(shape)
    dep_token = utils.generate_tensor_input(dep_token_shape, dtype, device)
    return plan.params["self"], plan.params["assert_msg"], dep_token


class FunctionalAssertScalarBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark measuring the dep_token copy volume."""

    def set_shapes(self, shape_file_path=None, default_shapes=None):
        defaults = [
            _check_dep_token_shape(shape)
            for shape in (default_shapes or DEFAULT_SCALES)
        ]
        if shape_file_path is None:
            self.shapes = defaults
        else:
            # The base loader resolves the operator name first and then the class name and
            # raises FileNotFoundError for an absent file or a YAML error for a malformed
            # one; neither is swallowed here, so a requested file is never replaced by the
            # defaults. Rows are dimension sequences, so a bare integer is rejected by the
            # validation below.
            super().set_shapes(shape_file_path, default_shapes=defaults)
        # Reject a requested row before any input tensor is built instead of silently
        # dropping it or substituting the defaults.
        self.shapes = [_check_dep_token_shape(shape) for shape in self.shapes]


@pytest.mark.functional_assert_scalar
def test__functional_assert_scalar():
    bench = FunctionalAssertScalarBenchmark(
        op_name="_functional_assert_scalar",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._functional_assert_scalar,
        gems_op=getattr(flag_gems, "_functional_assert_scalar", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
