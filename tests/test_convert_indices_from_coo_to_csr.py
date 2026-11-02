# Copyright 2026, The FlagOS Contributors.
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

"""Correctness tests for ``aten::_convert_indices_from_coo_to_csr``.

The operator consumes a rank-0/rank-1 integral COO row-index vector plus a row
count ``size`` and returns the CSR row pointers of length ``size + 1``:
``out[0] == 0`` and ``out[r + 1] == out[r] + count(r)``. Pointer dtype is int64
by default; ``out_int32=True`` selects int32.

Inputs are integral, never floating: the native operator has no float kernel, so
the rejected-dtype case is a negative test rather than a value-range case.
Sortedness is not validated natively (and is not validated by this
implementation either), so every positive case keeps its indices ascending.
``size`` may exceed the index count; the trailing rows simply stay empty.

The result is an integral pointer table whose values do not depend on the
device that produced them, so these cases are equally valid under
``utils.TO_CPU`` (the suite passes with ``--ref=cpu``); no skip is needed.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# CI derives the pytest marker from the implementation file name via the
# #6359 convention: the stripped marker collides with this operator's own
# registered id, so the derived marker takes the ``underscore_`` prefix.
# Declare both so ``pytest -m <marker>`` selects these tests from either
# entry point (same convention as the merged ``_indices_copy`` suite).


# Index dtypes the operator accepts. The 8-bit forms are added explicitly
# because the shared pool starts at int16; int64 is gated by the device
# capability the runtime publishes.
INDEX_DTYPES = [torch.int8, torch.uint8] + list(utils.ALL_INT_DTYPES)

# Row counts and index counts exercised on every dtype. Short and long index
# vectors and pointer tables that are wider or narrower than the index vector
# all map to distinct kernel paths (bisection trip count, trailing empty rows).
SIZE_CASES = [
    (0, 0),
    (0, 8),
    (1, 1),
    (17, 64),
    (64, 64),
    (64, 257),
    (4096, 4096),
    (65536, 17),
    (65536, 65536),
]


def _sorted_indices(dtype, nnz, size, seed=0):
    """Ascending row indices in ``[0, size)`` for the given dtype."""
    if nnz == 0:
        return torch.empty(0, dtype=dtype, device=flag_gems.device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bound = min(size, int(torch.iinfo(dtype).max) + 1)
    draw = torch.randint(0, bound, (nnz,), generator=gen, dtype=torch.int64)
    return draw.sort().values.to(dtype).to(flag_gems.device)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("nnz,size", SIZE_CASES)
@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_convert_indices_from_coo_to_csr(nnz, size, dtype):
    inp = _sorted_indices(dtype, nnz, size)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr(ref_inp, size)
    res_out = flag_gems._convert_indices_from_coo_to_csr(inp, size)

    assert res_out.dtype == torch.int64
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("out_int32", [False, True])
@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_convert_indices_from_coo_to_csr_out_int32_grid(out_int32, dtype):
    inp = _sorted_indices(dtype, 4096, 4096, seed=1)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr(
        ref_inp, 4096, out_int32=out_int32
    )
    res_out = flag_gems._convert_indices_from_coo_to_csr(inp, 4096, out_int32=out_int32)

    assert res_out.dtype == (torch.int32 if out_int32 else torch.int64)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("value,size", [(0, 1), (3, 4), (3, 32), (63, 64)])
def test_convert_indices_from_coo_to_csr_scalar_input(value, size):
    # A rank-0 index tensor is a valid input.
    inp = torch.tensor(value, dtype=torch.int32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr(ref_inp, size)
    res_out = flag_gems._convert_indices_from_coo_to_csr(inp, size)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.convert_indices_from_coo_to_csr
def test_convert_indices_from_coo_to_csr_offset_view():
    # A unit-stride view with a non-zero storage offset is honoured natively.
    # The kernel reads through the index stride, so the view is honoured too.
    # Under --ref=cpu the reference runs the native CPU kernel, which writes
    # past the output buffer for offset views and corrupts the heap (PyTorch
    # bug, reproducible outside this suite), so that pass is skipped.
    if utils.TO_CPU:
        pytest.skip("native CPU kernel corrupts the heap on offset views")
    size = 64
    base = (torch.arange(2 * size + 8, dtype=torch.int32)).to(flag_gems.device)
    inp = base.narrow(0, 7, size)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._convert_indices_from_coo_to_csr(ref_inp, size)
    res_out = flag_gems._convert_indices_from_coo_to_csr(inp, size)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.convert_indices_from_coo_to_csr_out
@pytest.mark.parametrize("out_int32", [False, True])
def test_convert_indices_from_coo_to_csr_out(out_int32):
    size = 64
    inp = _sorted_indices(torch.int32, 4 * size, size, seed=2)
    ref_inp = utils.to_reference(inp)
    out_dtype = torch.int32 if out_int32 else torch.int64

    res_out = torch.empty(size + 1, dtype=out_dtype, device=flag_gems.device)
    # The reference buffer follows the reference operand: under --ref=cpu the
    # native .out call needs all of its tensors on the same device.
    ref_out = torch.empty(size + 1, dtype=out_dtype, device=ref_inp.device)

    res_ret = flag_gems._convert_indices_from_coo_to_csr_out(
        inp, size, out_int32=out_int32, out=res_out
    )
    ref_ret = torch.ops.aten._convert_indices_from_coo_to_csr.out(
        ref_inp, size, out_int32=out_int32, out=ref_out
    )

    assert res_ret is res_out
    utils.gems_assert_equal(res_ret, ref_ret)


@pytest.mark.convert_indices_from_coo_to_csr
def test_convert_indices_from_coo_to_csr_out_offset_view():
    # The out buffer is a unit-stride window inside a sentinel-filled parent, so
    # a write outside the requested size + 1 elements shows up in the parent.
    size = 64
    guard = 4
    sentinel = -99
    inp = _sorted_indices(torch.int32, 4 * size, size, seed=3)
    ref_inp = utils.to_reference(inp)

    res_parent = torch.full(
        (size + 1 + 2 * guard,),
        sentinel,
        dtype=torch.int64,
        device=flag_gems.device,
    )
    res_buf = res_parent.narrow(0, guard, size + 1)
    ref_parent = torch.full(
        (size + 1 + 2 * guard,),
        sentinel,
        dtype=torch.int64,
        device=ref_inp.device,
    )
    ref_buf = ref_parent.narrow(0, guard, size + 1)

    res_ret = flag_gems._convert_indices_from_coo_to_csr_out(inp, size, out=res_buf)
    ref_ret = torch.ops.aten._convert_indices_from_coo_to_csr.out(
        ref_inp, size, out=ref_buf
    )

    assert res_ret is res_buf
    utils.gems_assert_equal(res_ret, ref_ret)
    utils.gems_assert_equal(res_parent, ref_parent)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (1, 1, 1, 1)])
def test_convert_indices_from_coo_to_csr_rejects_non_vector(shape):
    inp = torch.zeros(shape, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._convert_indices_from_coo_to_csr(inp, 8)


# No floating dtype has a kernel: the operator is integral-only.
@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize(
    "dtype",
    # Rejected-dtype matrix: no floating dtype has a kernel, so every entry here
    # is a negative case rather than a tested precision.
    [
        torch.float16,
        torch.float32,
        torch.bool,
    ]
    # bfloat16 is a rejection case (no kernel), not a tested precision.
    + ([torch.bfloat16] if utils.bf16_is_supported else [])
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else [])
    + ([torch.float8_e4m3fn, torch.float8_e5m2] if utils.fp8_is_supported else []),
)
def test_convert_indices_from_coo_to_csr_rejects_unsupported_dtype(dtype):
    inp = torch.zeros(4, dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._convert_indices_from_coo_to_csr(inp, 8)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("size", [-1, -4])
def test_convert_indices_from_coo_to_csr_rejects_negative_size(size):
    inp = torch.zeros(4, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._convert_indices_from_coo_to_csr(inp, size)


@pytest.mark.convert_indices_from_coo_to_csr
@pytest.mark.parametrize("size", [2.5, None, "4"])
def test_convert_indices_from_coo_to_csr_rejects_invalid_size(size):
    inp = torch.zeros(4, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(TypeError):
        flag_gems._convert_indices_from_coo_to_csr(inp, size)


@pytest.mark.convert_indices_from_coo_to_csr
def test_convert_indices_from_coo_to_csr_rejects_invalid_out_int32():
    inp = torch.zeros(4, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(TypeError):
        flag_gems._convert_indices_from_coo_to_csr(inp, 4, out_int32="x")


@pytest.mark.convert_indices_from_coo_to_csr
def test_convert_indices_from_coo_to_csr_rejects_out_dtype_mismatch():
    # The out overload requires the buffer dtype selected by out_int32.
    inp = torch.zeros(4, dtype=torch.int32, device=flag_gems.device)
    out = torch.empty(5, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._convert_indices_from_coo_to_csr_out(inp, 4, out=out)
