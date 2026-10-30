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

"""Focused tests for the Kunlunxin in-place foreach fast paths.

``tests/test_foreach_unary.py`` parameterizes all thirty foreach unary
operators and only runs when that file itself changes.  The Kunlunxin
overrides for ``_foreach_abs_``/``_foreach_round_``/``_foreach_sign_`` need an
entry point that CI runs whenever their implementation files change: the
markers ``foreach_abs_``/``foreach_round_``/``foreach_sign_`` below are derived
from the changed backend files by ``tools/test-op.sh``.

The cases cover both halves of the override:

* the unmasked direct-pointer fast path (homogeneous float list, >= 128
  elements, size divisible by the block picked in ``_foreach_direct``), and
* the fallback to the generic executor for every shape/layout/dtype the fast
  path refuses (small or non-divisible sizes, mixed shapes/dtypes, gappy or
  offset views, > 16 tensors, integral inputs).
"""

import logging

import pytest
import torch

import flag_gems

from .accuracy_utils import (
    gems_assert_close,
    gems_assert_equal,
    gems_log_logger,
    gems_log_prefix,
    to_reference,
)
from .conftest import QUICK_MODE

# The fast path keeps the input dtype; bf16 is included because the kernels
# store through ``.to(DT)`` rather than computing in fp16.
FLOAT_DTYPES = (
    [torch.float32] if QUICK_MODE else [torch.float16, torch.float32, torch.bfloat16]
)

# Sizes that take the unmasked fast path: ``_foreach_direct._pick_block`` picks
# a power-of-two block and the gate requires ``numel % block == 0``.
FAST_SIZES = [128] if QUICK_MODE else [128, 1024, 4096]


def _gems_op(name):
    """The FlagGems wrapper ATen dispatches ``_foreach_<name>_`` to.

    Calling it directly is what ``use_gems()`` would have dispatched to; CI's
    check_kernelgen_tests job rejects ``use_gems()`` inside test files.
    """
    return getattr(flag_gems, f"_foreach_{name}_")


def _torch_op(name):
    return getattr(torch, f"_foreach_{name}_")


def _sample(shape, dtype, device):
    # randn covers negatives (abs/sign) and non-integral values (round).
    return torch.randn(shape, dtype=dtype, device=device)


def _check_inplace(name, inp):
    """Run the reference and the FlagGems in-place op on equal inputs."""
    ref_inp = [to_reference(t.clone()) for t in inp]
    _torch_op(name)(ref_inp)

    ret = _gems_op(name)(inp)
    # The in-place schema returns ``()``; returning the list would make the
    # dispatcher reject the kernel.
    assert ret is None

    for got, want in zip(inp, ref_inp):
        assert got.dtype == want.dtype, f"{name}: {got.dtype} != {want.dtype}"
        if got.dtype.is_floating_point:
            gems_assert_close(got, want, got.dtype)
        else:
            gems_assert_equal(got, want)


def _check_fallback_cases(name):
    device = flag_gems.device
    dtype = torch.float32
    base = _sample((8, 8), dtype, device)
    wide = _sample((8, 16), dtype, device)
    cases = [
        [_sample((7,), dtype, device)],  # below the 128-element gate
        [_sample((127,), dtype, device)],  # below the 128-element gate
        [_sample((129,), dtype, device)],  # not divisible by the block
        [_sample((4, 5), dtype, device), _sample((3,), dtype, device)],
        [wide[:, ::2]],  # gappy view
        [base[2:6]],  # non-zero storage offset
        [base.t()],  # dense but transposed
        [_sample((64,), dtype, device) for _ in range(17)],  # > MAX_DIRECT
        [
            _sample((64,), torch.float32, device),
            _sample((64,), torch.float16, device),
        ],  # mixed dtypes
    ]
    for inp in cases:
        _check_inplace(name, inp)


def _check_writes_through_view(name):
    """In-place on a gappy view must land in the original storage."""
    storage = _sample((8, 16), torch.float32, flag_gems.device)
    ref_storage = to_reference(storage.clone())

    _torch_op(name)([ref_storage[:, ::2]])
    _gems_op(name)([storage[:, ::2]])

    # Both the written half and the untouched half must match.
    gems_assert_close(storage, ref_storage, storage.dtype)


def _check_liveness(name, caplog):
    """The in-place registration must emit the vendor debug record."""
    if flag_gems.device != "cuda":
        return
    want = f"{gems_log_prefix(_gems_op(name))} _FOREACH_{name.upper()}_"
    logger_name = gems_log_logger(_gems_op(name))

    with caplog.at_level(logging.DEBUG, logger=logger_name):
        # Negative control: the ATen reference must not emit the record.
        caplog.clear()
        _torch_op(name)([_sample((8,), torch.float32, flag_gems.device)])
        assert want not in caplog.text, "negative control fired: probe proves nothing"

        caplog.clear()
        _gems_op(name)([_sample((8,), torch.float32, flag_gems.device)])
        assert want in caplog.text, f"dead registration for {want}"


# ---------------------------------------------------------------------------
# abs
# ---------------------------------------------------------------------------


@pytest.mark.foreach_abs_
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("size", FAST_SIZES)
def test_foreach_abs_inplace_fast_path(dtype, size):
    _check_inplace("abs", [_sample((size,), dtype, flag_gems.device) for _ in range(3)])


@pytest.mark.foreach_abs_
def test_foreach_abs_inplace_fallback_cases():
    _check_fallback_cases("abs")


@pytest.mark.foreach_abs_
def test_foreach_abs_inplace_int_fallback():
    inp = [torch.randint(-5, 6, (128,), dtype=torch.int32, device=flag_gems.device)]
    _check_inplace("abs", inp)


@pytest.mark.foreach_abs_
def test_foreach_abs_inplace_writes_through_view():
    _check_writes_through_view("abs")


@pytest.mark.foreach_abs_
def test_foreach_abs_inplace_registration_is_live(caplog):
    _check_liveness("abs", caplog)


@pytest.mark.foreach_abs_
def test_foreach_abs_inplace_rejects_empty_list():
    with pytest.raises(RuntimeError):
        _gems_op("abs")([])


# ---------------------------------------------------------------------------
# round
# ---------------------------------------------------------------------------


@pytest.mark.foreach_round_
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("size", FAST_SIZES)
def test_foreach_round_inplace_fast_path(dtype, size):
    _check_inplace(
        "round", [_sample((size,), dtype, flag_gems.device) for _ in range(3)]
    )


@pytest.mark.foreach_round_
def test_foreach_round_inplace_half_to_even():
    # 128 elements take the fast path; repeat keeps the exact tie values.
    ties = torch.tensor(
        [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, -3.5, 3.5],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    _check_inplace("round", [ties.repeat(16)])


@pytest.mark.foreach_round_
def test_foreach_round_inplace_large_magnitude():
    # Values around the 2**22 / 2**23 range boundaries of the magic-number
    # construction, including odd integers and half-integers where a plain
    # ``(x + 1.5*2**23) - 1.5*2**23`` would round incorrectly.
    vals = torch.tensor(
        [
            4194303.5,
            4194304.0,
            4194304.5,
            4194305.0,
            4194305.5,
            8388607.5,
            8388608.0,
            8388609.0,
            -4194303.5,
            -4194304.0,
            -4194304.5,
            -4194305.0,
            -4194305.5,
            -8388607.5,
            -8388608.0,
            -8388609.0,
        ],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    _check_inplace("round", [vals.repeat(8)])  # 128 elements -> fast path
    _check_inplace("round", [vals.repeat(4)])  # 64 elements -> fallback


@pytest.mark.foreach_round_
def test_foreach_round_inplace_fallback_cases():
    _check_fallback_cases("round")


@pytest.mark.foreach_round_
def test_foreach_round_inplace_writes_through_view():
    _check_writes_through_view("round")


@pytest.mark.foreach_round_
def test_foreach_round_inplace_registration_is_live(caplog):
    _check_liveness("round", caplog)


@pytest.mark.foreach_round_
def test_foreach_round_inplace_rejects_empty_list():
    with pytest.raises(RuntimeError):
        _gems_op("round")([])


# ---------------------------------------------------------------------------
# sign
# ---------------------------------------------------------------------------


@pytest.mark.foreach_sign_
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("size", FAST_SIZES)
def test_foreach_sign_inplace_fast_path(dtype, size):
    _check_inplace(
        "sign", [_sample((size,), dtype, flag_gems.device) for _ in range(3)]
    )


@pytest.mark.foreach_sign_
def test_foreach_sign_inplace_fallback_cases():
    _check_fallback_cases("sign")


@pytest.mark.foreach_sign_
def test_foreach_sign_inplace_int_fallback():
    inp = [torch.randint(-5, 6, (128,), dtype=torch.int32, device=flag_gems.device)]
    _check_inplace("sign", inp)


@pytest.mark.foreach_sign_
def test_foreach_sign_inplace_writes_through_view():
    _check_writes_through_view("sign")


@pytest.mark.foreach_sign_
def test_foreach_sign_inplace_registration_is_live(caplog):
    _check_liveness("sign", caplog)


@pytest.mark.foreach_sign_
def test_foreach_sign_inplace_rejects_empty_list():
    with pytest.raises(RuntimeError):
        _gems_op("sign")([])
