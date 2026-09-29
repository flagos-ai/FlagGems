# Copyright 2025 The FlagGems Authors.
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

_EXPLICIT_DTYPE = "explicit_dtype"
_EXPLICIT_LAYOUT = "explicit_layout"
_VALID_TAGS = (None, _EXPLICIT_DTYPE, _EXPLICIT_LAYOUT)
# The explicit-dtype workload overrides the output dtype, so an fp16/bf16 source
# allocates a float32 buffer instead of the inherited one.
_OVERRIDE_DTYPE = torch.float32

# (source geometry, requested size, tag).  The source geometry is what the case
# allocates; the requested size is the new_zeros output shape.  The last three
# rows exercise an unequal source/output geometry and explicit output
# dtype/layout parameters.
NEW_ZEROS_CASES = [
    ((256,), (256,), None),
    ((1024, 1024), (1024, 1024), None),
    ((4096, 4096), (4096, 4096), None),
    ((20, 320, 15), (20, 320, 15), None),
    ((64, 512, 512), (64, 512, 512), None),
    ((16, 128, 64, 60), (16, 128, 64, 60), None),
    ((64, 512, 512), (1024, 1024), None),
    ((1024, 1024), (1024, 1024), _EXPLICIT_DTYPE),
    ((1024, 1024), (1024, 1024), _EXPLICIT_LAYOUT),
]

# bf16 is not allocated on every backend; the gate is static so listing and
# execution share exactly the same dtype list.
_BENCH_DTYPES = [
    d
    for d in consts.FLOAT_DTYPES
    if d != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _describe_case(descriptor):
    # Validate one (source, size, tag) descriptor and return its normalised form.
    # The metadata planner (_case_fn) and the input builder both call this, so a
    # malformed or unknown-tag row fails during --list-cases instead of being
    # listed and only rejected when the same case is executed or replayed.
    if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 3:
        raise ValueError(
            f"new_zeros case descriptor must be (source, size, tag), got {descriptor!r}"
        )
    source, size, tag = descriptor
    if not isinstance(source, (tuple, list)):
        raise ValueError(
            f"new_zeros source geometry must be a sequence, got {source!r}"
        )
    if not isinstance(size, (tuple, list)):
        raise ValueError(f"new_zeros size must be a sequence, got {size!r}")
    # bool extents are valid for the native schema, which coerces True to 1, so
    # they are normalised to the integer the call really uses.
    if any(not isinstance(e, int) or int(e) < 0 for e in list(source) + list(size)):
        raise ValueError(
            f"new_zeros extents must be non-negative integers, got source={source!r} size={size!r}"
        )
    if tag not in _VALID_TAGS:
        raise ValueError(
            f"unknown new_zeros case tag {tag!r}, expected one of {_VALID_TAGS}"
        )
    return tuple(int(e) for e in source), tuple(int(e) for e in size), tag


def _case_fn(shape, dtype):
    source, size, tag = _describe_case(shape)
    params = {"size": list(size)}
    if tag == _EXPLICIT_DTYPE:
        params["dtype"] = str(_OVERRIDE_DTYPE)
        params["source_dtype"] = str(dtype)
    elif tag == _EXPLICIT_LAYOUT:
        params["layout"] = str(torch.strided)
    yield base.BenchmarkCasePlan(
        shape={"input": list(source)},
        params=params,
        builder_args=(source, size, tag),
    )


def _build_inputs_fn(plan, dtype, device):
    source, size, tag = _describe_case(plan.builder_args)
    inp = utils.generate_tensor_input(source, dtype, device)
    kwargs = {}
    if tag == _EXPLICIT_DTYPE:
        kwargs["dtype"] = _OVERRIDE_DTYPE
    elif tag == _EXPLICIT_LAYOUT:
        kwargs["layout"] = torch.strided
    return inp, size, kwargs


class NewZerosBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # new_zeros has no entry in the shared shape files: keep the scales above
        # as the defaults while still honouring a caller supplied --shape_file.
        super().set_shapes(shape_file_path, default_shapes=NEW_ZEROS_CASES)


@pytest.mark.new_zeros
def test_new_zeros():
    bench = NewZerosBenchmark(
        op_name="new_zeros",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.new_zeros,
        gems_op=getattr(flag_gems, "new_zeros", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
