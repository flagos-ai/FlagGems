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

# aten::is_set_to(Tensor self, Tensor tensor) -> bool inspects allocation
# metadata only, so its cost follows the operand rank and the storage-view
# relation rather than the element count. Every shape of the shared benchmark
# grid (default, comprehensive and caller shape files) is expanded into the
# relations that shape can express. Fixtures use torch.empty because neither the
# operator nor this file reads an element.

_RELATIONS = (
    "same_object",
    "detached",
    "same_geometry_view",
    "storage_set_twin",
    "clone",
    "narrowed",
    "strided_slice",
    "transposed",
    "expanded",
)

# Static capability flags, read at import time: no tensor is allocated and no
# operator is called during collection. An unmapped dtype is a baseline type the
# backend always handles.
_CAPABILITY_FLAGS = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _CAPABILITY_FLAGS.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


BENCH_DTYPES = [
    dtype
    for dtype in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
        torch.float64,
    )
    if _dtype_supported(dtype)
]


def _relation_applies(relation, shape):
    """Rank/extent restrictions of the relation built by _build_pair."""
    if relation == "narrowed":
        return len(shape) >= 1 and shape[0] >= 2
    if relation == "strided_slice":
        return len(shape) >= 2 and shape[1] >= 2
    if relation == "transposed":
        return len(shape) >= 2
    return True


def _other_shape(relation, shape):
    """Shape of the second operand, computed without allocating anything."""
    if relation in (
        "same_object",
        "detached",
        "same_geometry_view",
        "storage_set_twin",
        "clone",
    ):
        return shape
    if relation == "narrowed":
        return (shape[0] - 1,) + tuple(shape[1:])
    if relation == "strided_slice":
        return (shape[0], shape[1] - shape[1] // 2) + tuple(shape[2:])
    if relation == "transposed":
        return (shape[-1],) + tuple(shape[1:-1]) + (shape[0],)
    return (2,) + tuple(shape)


def _build_pair(relation, inp):
    """Return two operands describing relation over the storage of inp."""
    if relation == "same_object":
        return inp, inp
    if relation == "detached":
        return inp, inp.detach()
    if relation == "same_geometry_view":
        return inp, inp.as_strided(inp.shape, inp.stride(), inp.storage_offset())
    if relation == "storage_set_twin":
        twin = torch.empty(0, dtype=inp.dtype, device=inp.device)
        twin.set_(inp.untyped_storage(), inp.storage_offset(), inp.size(), inp.stride())
        return inp, twin
    if relation == "clone":
        # Same geometry in separate storage; no payload is read, so an
        # uninitialized allocation carries the same relation a clone would.
        return inp, torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)
    if relation == "narrowed":
        return inp, inp[1:]
    if relation == "strided_slice":
        return inp, inp[:, ::2]
    if relation == "transposed":
        return inp, inp.transpose(0, -1)
    return inp, inp.unsqueeze(0).expand((2,) + tuple(inp.shape))


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    for relation in _RELATIONS:
        if not _relation_applies(relation, shape):
            continue
        yield base.BenchmarkCasePlan(
            shape={"self": list(shape), "other": list(_other_shape(relation, shape))},
            params={"relation": relation},
            builder_args=(relation, shape),
        )


def _build_inputs_fn(plan, dtype, device):
    relation, shape = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    return (*_build_pair(relation, inp), {})


@pytest.mark.is_set_to
def test_is_set_to():
    bench = base.GenericBenchmark(
        op_name="is_set_to",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_set_to,
        gems_op=getattr(flag_gems, "is_set_to", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
