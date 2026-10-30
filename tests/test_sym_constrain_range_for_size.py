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
from flag_gems.ops.sym_constrain_range_for_size import (
    sym_constrain_range_for_size as gems_impl,
)

from . import accuracy_utils as utils

# aten::sym_constrain_range_for_size(Scalar size, *, int? min=None, int? max=None) -> ()
# is a host-side / compile-graph guard for symbolic sizes: it checks that the
# size lies in [min, max] and returns nothing. It has no tensor argument and no
# tensor result, so the regular spec's tensor grids (value ranges, shape levels,
# broadcast, backward, dtype lists) do not apply. Covered instead: the Scalar
# value domain, the four bound call forms, exact int64 boundaries, the
# keyword-only signature, and rejection of invalid sizes and bounds.

_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63)

# Scalar sizes across the accepted domain, including the float and bool forms
# Scalar also accepts.
_SIZE_VALUES = [
    0,
    1,
    2,
    3,
    4,
    5,
    7,
    42,
    256,
    1024,
    4096,
    65536,
    2**31,
    2**62,
    _INT64_MAX,
    3.9,
    2.999,
    -0.5,
    True,
    False,
]

# The four bound call forms: neither, min only, max only, both. A None bound is
# omitted from the call so the schema default is exercised.
_BOUND_FORMS = [
    (None, None),
    (0, None),
    (None, _INT64_MAX),
    (0, _INT64_MAX),
    (-8, None),
]

# Exact boundaries at both ends of the accepted domain.
_BOUNDARY_ROWS = [
    (3, 3, 3),
    (3, 0, 3),
    (0, 0, 3),
    (0, -4, 3),
    (4, 3, 4),
    (5, 4, 5),
    (10, 10, 10),
    (1024, 0, 1024),
    (4096, 1, 4096),
    (65536, 2, 65536),
    (-1, -1, 3),
    (-5, -10, 3),
    (-3.9, -10, 3),
    (_INT64_MAX, 0, _INT64_MAX),
    (_INT64_MAX, _INT64_MAX, _INT64_MAX),
    (2**62, 2**62 - 1, 2**62),
    (_INT64_MIN + 1, _INT64_MIN, 3),
]

# Quick keeps a small scalar grid that still spans every bound call form.
_ACCEPTED_ROWS = [
    (size, min_value, max_value)
    for size in _SIZE_VALUES
    for min_value, max_value in _BOUND_FORMS
] + _BOUNDARY_ROWS


def _bound_kwargs(min_value, max_value):
    """Pass only the bounds that are actually supplied."""
    return {
        name: value
        for name, value in (("min", min_value), ("max", max_value))
        if value is not None
    }


@pytest.mark.sym_constrain_range_for_size
@pytest.mark.parametrize("size, min_value, max_value", _ACCEPTED_ROWS)
def test_sym_constrain_range_for_size(size, min_value, max_value):
    kwargs = _bound_kwargs(min_value, max_value)

    ref_out = torch.ops.aten.sym_constrain_range_for_size(size, **kwargs)
    res_out = flag_gems.sym_constrain_range_for_size(size, **kwargs)

    # void operator: neither call may produce a result
    assert (res_out, ref_out) == (None, None)


# Rejected calls: only "raises" is asserted, since the error class is an
# implementation detail.
_REJECTED_ROWS = [
    # size outside the requested range, or outside the default 0 .. int64 max
    (5, 6, None),
    (5, 0, 4),
    (200, 0, 100),
    (0, 1, None),
    (-1, 0, None),
    (-3.9, None, None),
    (2, 3, 10),
    (4096, 0, 4095),
    (-5, -3, 3),
    (2**62, None, 2**62 - 1),
    (_INT64_MAX, None, _INT64_MAX - 1),
    (50 + 0j, 0, 10),
    # max, when given, must be greater than 2
    (5, None, 2),
    (5, None, 1),
    (5, None, 0),
    (5, None, -1),
    (0, None, 2),
    (5, 0, 2),
    (5, 10, 2),
    (5, None, True),
    # min must be less than or equal to max
    (5, 10, 3),
    (5, 5, 4),
    (3, 4, 3),
    (100, 4, 3),
    (10, 2**62, 2**62 - 1),
    # min/max are schema ints; every row is also out of range under a lenient
    # implementation, so dropping the bounds cannot turn it into a pass
    (0, 1.0, None),
    (5, 0.5, 10),
    (8, 0, 3.0),
    (5, 0, 2.5),
    (5, "0", 10),
    (5, 0, "10"),
    (5, 0, "3"),
    # size must be a number: strings, containers, out-of-int64 and non-finite
    # values are all rejected
    ("5", None, None),
    (None, None, None),
    ([5], None, None),
    ((5,), None, None),
    (2**63, None, None),
    (float("nan"), None, None),
    (float("inf"), None, None),
    (float("-inf"), None, None),
    (1e30, None, None),
    (5 + 1j, None, None),
]


@pytest.mark.sym_constrain_range_for_size
@pytest.mark.parametrize("size, min_value, max_value", _REJECTED_ROWS)
def test_sym_constrain_range_for_size_rejects_invalid_arguments(
    size, min_value, max_value
):
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.sym_constrain_range_for_size(
            size, **_bound_kwargs(min_value, max_value)
        )


@pytest.mark.sym_constrain_range_for_size
def test_sym_constrain_range_for_size_rejects_positional_bounds():
    # min/max are keyword-only in the schema.
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.sym_constrain_range_for_size(5, 0, 10)


@pytest.mark.sym_constrain_range_for_size
def test_sym_constrain_range_for_size_requires_size():
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.sym_constrain_range_for_size()


@pytest.mark.sym_constrain_range_for_size
@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((), torch.int64),
        ((1,), torch.int32),
        ((), torch.float32),
        ((2,), torch.int64),
    ],
)
def test_sym_constrain_range_for_size_rejects_tensor_size(shape, dtype):
    # A tensor is not a Scalar, whatever it is shaped and whatever it holds.
    size = torch.full(shape, 5, dtype=dtype, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.sym_constrain_range_for_size(size)


@pytest.mark.sym_constrain_range_for_size
def test_sym_constrain_range_for_size_runs_triton_kernel(caplog):
    # The wrapper must reach its Triton kernel rather than decide the range on
    # the host, so assert the implementation's own debug log was emitted.
    with caplog.at_level("DEBUG", logger=utils.gems_log_logger(gems_impl)):
        res_out = gems_impl(5, min=0, max=10)

    prefix = utils.gems_log_prefix(gems_impl)
    assert f"{prefix} SYM_CONSTRAIN_RANGE_FOR_SIZE" in caplog.text
    assert res_out is None


@pytest.mark.sym_constrain_range_for_size
def test_sym_constrain_range_for_size_dispatch_is_reachable():
    # Registration-path check. This operator takes a Scalar and no tensor, so it
    # carries no device key: on this build ATen registers it as
    # CompositeExplicitAutograd and a backend-key registration is never selected
    # (measured: a CUDA-key impl that always raises is not entered). The entry in
    # _FULL_CONFIG therefore also registers the alias key, and the sentinel below
    # proves `torch.ops.aten` routes into the registered implementation instead
    # of falling through to the native kernel.
    hits = []

    def sentinel(size, *, min=None, max=None):
        hits.append((size, min, max))
        return flag_gems.sym_constrain_range_for_size(size, min=min, max=max)

    lib = torch.library.Library("aten", "IMPL")
    try:
        lib.impl("sym_constrain_range_for_size", sentinel, "CompositeExplicitAutograd")

        assert torch.ops.aten.sym_constrain_range_for_size(5, min=0, max=10) is None
        with pytest.raises(RuntimeError):
            torch.ops.aten.sym_constrain_range_for_size(50, min=0, max=10)
    finally:
        lib._destroy()

    assert hits == [(5, 0, 10), (50, 0, 10)]
