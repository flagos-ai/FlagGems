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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# aten::reshape_as(self, other) reads only the shape of 'other', so a plain shape
# entry expands into three plans (contiguous, transposed and stride-0 expanded
# storage) while an explicit descriptor yields exactly the reported
# shape / target / layout. --list-cases and execution both consume the plans
# produced here; every descriptor field is validated and a target is checked
# against the reported numel before any tensor is allocated.
RESHAPE_AS_SHAPES = [
    (256,),
    (2, 19, 7),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64),
    (64, 128, 256),
    (16, 7, 57, 32),
    (4096, 4096),
]

# Numel-preserving target for each input scale above. Only its shape is read.
_TARGET_SHAPES = {
    (256,): (16, 16),
    (2, 19, 7): (7, 38),
    (1024, 1024): (1048576,),
    (20, 320, 15): (15, 320, 20),
    (16, 128, 64): (64, 128, 16),
    (64, 128, 256): (256, 128, 64),
    (16, 7, 57, 32): (32, 57, 7, 16),
    (4096, 4096): (16777216,),
}

# Static capability flags, read while the module is imported: no tensor is
# allocated and no operator is called at collection time. A dtype outside the map
# is a baseline type the backend always handles.
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


BENCH_DTYPES = [dtype for dtype in consts.FLOAT_DTYPES if _dtype_supported(dtype)]

_LAYOUTS = ("asis", "transposed", "expanded")
_DESCRIPTOR_FIELDS = frozenset({"shape", "input", "target", "layout", "base"})


def _checked_extent(extent):
    if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
        raise ValueError("invalid shape extent " + repr(extent))
    return extent


def _checked_shape(shape):
    # A scalar, an empty or a zero-sized shape stays exactly as requested;
    # booleans, non-integers, negative extents and non-sequences are rejected
    # here, before any tensor is allocated or any case is listed.
    try:
        extents = tuple(shape)
    except TypeError:
        raise ValueError("a shape must be a sequence of extents") from None
    return tuple(_checked_extent(extent) for extent in extents)


def _swap_last_two(shape):
    if len(shape) < 2:
        raise ValueError("a transposed layout needs at least two dimensions")
    return shape[:-2] + (shape[-1], shape[-2])


def _checked_expansion(base_shape, shape):
    if len(base_shape) != len(shape):
        raise ValueError("an expanded layout needs matching ranks")
    for base_extent, extent in zip(base_shape, shape):
        if base_extent != extent and base_extent != 1:
            raise ValueError("shape is not an expansion of the base shape")
    return base_shape


def _target_shape(shape):
    # Listed scales keep the explicit rank change; a caller shape that is not
    # listed falls back to a validated flatten, which preserves the numel
    # contract without inventing a rank.
    if shape in _TARGET_SHAPES:
        return _TARGET_SHAPES[shape]
    return _checked_shape((math.prod(shape),))


def _checked_target(shape, target):
    if target is None:
        return _target_shape(shape)
    target = _checked_shape(target)
    if math.prod(target) != math.prod(shape):
        raise ValueError("target numel differs from input numel")
    return target


def _is_descriptor(spec):
    if isinstance(spec, dict):
        return True
    return bool(spec) and isinstance(spec[0], (list, tuple))


def _normalize_descriptor(spec):
    # An explicit descriptor is a dict with shape/input, target, layout and base
    # keys, or a (shape, target[, layout[, base]]) sequence, at most four fields.
    # Every part is validated: unknown keys, conflicting shape/input fields,
    # excess arity and a base shape on a layout that does not expand are rejected
    # instead of being silently ignored.
    if isinstance(spec, dict):
        unknown = frozenset(spec) - _DESCRIPTOR_FIELDS
        if unknown:
            raise ValueError("unknown descriptor field " + repr(sorted(unknown)))
        if "shape" in spec and "input" in spec:
            if _checked_shape(spec["shape"]) != _checked_shape(spec["input"]):
                raise ValueError("conflicting shape and input fields")
            shape = spec["shape"]
        elif "shape" in spec:
            shape = spec["shape"]
        elif "input" in spec:
            shape = spec["input"]
        else:
            raise ValueError("a descriptor needs a shape or input field")
        target = spec.get("target")
        layout = spec.get("layout", "asis")
        base_shape = spec.get("base")
    else:
        if len(spec) > 4:
            raise ValueError("a descriptor takes at most four fields")
        shape = spec[0]
        target = spec[1] if len(spec) > 1 else None
        layout = spec[2] if len(spec) > 2 else "asis"
        base_shape = spec[3] if len(spec) > 3 else None

    shape = _checked_shape(shape)
    if layout not in _LAYOUTS:
        raise ValueError("unsupported layout " + repr(layout))
    if base_shape is not None and layout != "expanded":
        raise ValueError("a base shape only applies to an expanded layout")
    if layout == "transposed":
        storage = _swap_last_two(shape)
        expand_shape = None
    elif layout == "expanded":
        if base_shape is None:
            raise ValueError("an expanded layout needs an explicit base shape")
        storage = _checked_expansion(_checked_shape(base_shape), shape)
        expand_shape = shape
    else:
        storage = shape
        expand_shape = None
    return shape, _checked_target(shape, target), layout, storage, expand_shape


def _descriptor_plan(spec):
    shape, target, layout, storage, expand_shape = _normalize_descriptor(spec)
    return base.BenchmarkCasePlan(
        shape={"input": list(shape), "target": list(target)},
        params={"layout": layout},
        builder_args=(storage, layout, expand_shape, target),
    )


def _generated_plans(shape):
    numel = math.prod(shape)
    target = _target_shape(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape), "target": list(target)},
        params={"layout": "asis"},
        builder_args=(shape, "asis", None, target),
    )
    if len(shape) >= 2:
        yield base.BenchmarkCasePlan(
            shape={"input": list(_swap_last_two(shape)), "target": list(target)},
            params={"layout": "transposed"},
            builder_args=(shape, "transposed", None, target),
        )
    yield base.BenchmarkCasePlan(
        shape={"input": [2, numel], "target": [2 * numel]},
        params={"layout": "expanded"},
        builder_args=((1, numel), "expanded", (2, numel), (2 * numel,)),
    )


def _case_fn(shape, dtype):
    del dtype
    if _is_descriptor(shape):
        yield _descriptor_plan(shape)
        return
    shape = _checked_shape(shape)
    yield from _generated_plans(shape)


def _build_inputs_fn(plan, dtype, device):
    storage, layout, expand_shape, target = plan.builder_args
    inp = utils.generate_tensor_input(storage, dtype, device)
    if layout == "transposed":
        inp = inp.transpose(-1, -2)
    elif layout == "expanded":
        # Tuple form keeps the scalar expansion (shape=(), base=()) valid.
        inp = inp.expand(tuple(expand_shape))
    other = torch.empty(target, dtype=dtype, device=device)
    return inp, other, {}


class ReshapeAsBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=RESHAPE_AS_SHAPES)

    def set_more_shapes(self):
        # The three layouts above already exercise the view and the materializing
        # branches, so no additional shape set is contributed.
        return []


@pytest.mark.reshape_as
def test_reshape_as():
    bench = ReshapeAsBenchmark(
        op_name="reshape_as",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.reshape_as,
        gems_op=getattr(flag_gems, "reshape_as", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
