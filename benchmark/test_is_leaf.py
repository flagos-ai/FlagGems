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

from . import base

# aten::is_leaf reads autograd metadata only: no element value reaches the
# operator and nothing is written back. The shared default shape grid, the
# COMPREHENSIVE additions and the caller's --shape-file support are used
# unchanged; inputs come from torch.empty because contents are never read, and
# allocation stays outside the timed region.
#
# States: 'leaf' exists for every supported dtype. The requires-grad leaf, the
# non-leaf view and its detached copy need autograd tracking, which PyTorch only
# accepts for floating/complex dtypes ('only Tensors of floating point dtype can
# require gradients' for int/bool). The non-leaf is a view of the tracked tensor
# -- a real ViewBackward node, no elementwise kernel -- so it is available for
# FP8 as well. The view preserves the requested shape and is valid for every grid and caller shape.

_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


BENCH_DTYPES = [
    dtype
    for dtype in (
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
        torch.float64,
        torch.complex128,
    )
    if _dtype_supported(dtype)
]

_TRACKED_DTYPES = tuple(
    dtype for dtype in BENCH_DTYPES if dtype.is_floating_point or dtype.is_complex
)

LEAF_STATE = "leaf"
TRACKED_STATES = ("grad_leaf", "nonleaf_view", "detached_view")


def _case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"state": LEAF_STATE},
        builder_args=(shape, LEAF_STATE),
    )
    if dtype in _TRACKED_DTYPES:
        for state in TRACKED_STATES:
            yield base.BenchmarkCasePlan(
                shape={"input": list(shape)},
                params={"state": state},
                builder_args=(shape, state),
            )


def _build_inputs_fn(plan, dtype, device):
    shape, state = plan.builder_args
    # Metadata-only fixture: is_leaf reads no element, so an uninitialized
    # allocation is enough and its contents are never compared.
    inp = torch.empty(shape, dtype=dtype, device=device)
    if state == LEAF_STATE:
        return inp, {}
    tracked = inp.requires_grad_(True)
    if state == "grad_leaf":
        return tracked, {}
    view = tracked.view(shape)
    if state == "nonleaf_view":
        return view, {}
    return view.detach(), {}


@pytest.mark.is_leaf
def test_is_leaf():
    bench = base.GenericBenchmark(
        op_name="is_leaf",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_leaf,
        gems_op=getattr(flag_gems, "is_leaf", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
