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

from . import accuracy_utils as utils


@pytest.mark.msort
@pytest.mark.parametrize("rows", [17, 1025])
@pytest.mark.parametrize(
    "dtype,int_dtype,bits",
    [
        (
            torch.float32,
            torch.int32,
            [0x7FC00001, 0x7FC00031, -0x003FFFFF, -0x003FFFCD],
        ),
        (
            torch.float64,
            torch.int64,
            [
                0x7FF8000000000001,
                0x7FF8000000000031,
                -0x0007FFFFFFFFFFFF,
                -0x0007FFFFFFFFFFCD,
            ],
        ),
    ],
)
def test_msort_preserves_nan_payloads(rows, dtype, int_dtype, bits):
    inp = torch.randn(rows, 3, dtype=dtype, device=flag_gems.device)
    payloads = torch.tensor(bits, dtype=int_dtype, device=inp.device).view(dtype)
    inp[:4] = payloads[:, None]
    inp[4] = float("inf")
    inp[5] = -float("inf")
    # Older CUDA radix-sort implementations order negative NaNs before -inf.
    # CPU ATen provides the intended NaNs-last ordering while preserving bits.
    reference = torch.msort(utils.to_reference(inp).cpu())
    result = flag_gems.msort(inp).cpu()
    torch.testing.assert_close(result, reference, atol=0, rtol=0, equal_nan=True)
    for col in range(3):
        actual_nan = result[:, col][torch.isnan(result[:, col])].view(int_dtype)
        expected_nan = reference[:, col][torch.isnan(reference[:, col])].view(int_dtype)
        torch.testing.assert_close(
            torch.sort(actual_nan).values,
            torch.sort(expected_nan).values,
            atol=0,
            rtol=0,
        )
        assert int(torch.signbit(result[:, col][-4:]).sum()) == 2


@pytest.mark.msort_out
@pytest.mark.parametrize("rows", [17, 512, 513, 1025])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("view_alias", [False, True])
def test_msort_out_alias_across_sort_paths(rows, dtype, view_alias):
    # Many columns and rows force multiple programs in both sorting paths.
    inp = _make_input((rows, 129), dtype)
    reference_input = utils.to_reference(inp).clone()
    reference = torch.msort(reference_input, out=reference_input)
    out = inp.view_as(inp) if view_alias else inp
    pointer, strides = out.data_ptr(), out.stride()
    result = flag_gems.msort_out(inp, out=out)
    assert result is out and result.data_ptr() == pointer
    assert result.stride() == strides
    utils.gems_assert_equal(result, reference)


def _make_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_gems.device)
    if not dtype.is_floating_point:
        return torch.randint(-100, 101, shape, dtype=dtype, device=flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


@pytest.mark.msort
@pytest.mark.parametrize("shape", [(1,), (7,), (8, 17), (63, 129), (257, 7), (3, 5, 7)])
@pytest.mark.parametrize(
    "dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + [torch.bool]
)
def test_msort(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.msort(ref_inp)

    result = flag_gems.msort(inp)

    assert result.dtype == inp.dtype
    assert result.stride() == inp.stride()
    utils.gems_assert_equal(result, ref_out)


@pytest.mark.msort
def test_msort_noncontiguous_empty_scalar_and_special_values():
    inp = torch.randn((19, 7), device=flag_gems.device).T
    ref_out = torch.msort(utils.to_reference(inp))
    result = flag_gems.msort(inp)
    assert result.stride() == inp.stride()
    utils.gems_assert_equal(result, ref_out)

    for shape in [(), (0,), (0, 7), (3, 0)]:
        inp = torch.empty(shape, device=flag_gems.device)
        result = flag_gems.msort(inp)
        assert result.shape == inp.shape
        assert result.stride() == inp.stride()

    inp = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 2.0, -3.0],
        device=flag_gems.device,
    )
    ref_out = torch.msort(utils.to_reference(inp))
    result = flag_gems.msort(inp)
    utils.gems_assert_equal(result, ref_out, equal_nan=True)

    # Exercise the large-first-dimension radix fallback.
    inp = torch.randn((1025, 3), device=flag_gems.device)
    inp[0, 0] = -float("nan")
    inp[1, 0] = float("nan")
    inp[2, 0] = -float("inf")
    inp[3, 0] = float("inf")
    # CUDA's own radix path can expose the same signed-NaN ordering bug;
    # use the CPU ATen result as the semantic reference for this regression.
    ref_out = torch.msort(inp.cpu())
    torch.testing.assert_close(
        flag_gems.msort(inp).cpu(), ref_out, atol=0, rtol=0, equal_nan=True
    )


@pytest.mark.msort
@pytest.mark.parametrize("shape", [(), (0,), (1,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_msort_rejects_complex(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        torch.msort(utils.to_reference(inp))
    with pytest.raises(RuntimeError):
        flag_gems.msort(inp)


@pytest.mark.msort_out
@pytest.mark.parametrize("shape", [(8, 17), (63, 129), (257, 7), (3, 5, 7)])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_msort_out(shape, dtype):
    inp = _make_input(shape, dtype)
    ref_out = torch.msort(utils.to_reference(inp))
    out = torch.empty_like(inp)
    ptr = out.data_ptr()

    result = flag_gems.msort_out(inp, out=out)

    assert result is out
    assert result.data_ptr() == ptr
    utils.gems_assert_equal(result, ref_out)


@pytest.mark.msort_out
def test_msort_out_resize_noncontiguous_alias_and_dtype_error():
    inp = torch.randn((19, 7), device=flag_gems.device).T
    ref_out = torch.msort(utils.to_reference(inp))
    out = torch.empty((1,), device=flag_gems.device)
    result = flag_gems.msort_out(inp, out=out)
    assert result is out
    assert result.shape == inp.shape
    utils.gems_assert_equal(result, ref_out)

    alias = torch.randn((31, 13), device=flag_gems.device)
    ref_alias = torch.msort(utils.to_reference(alias))
    result = flag_gems.msort_out(alias, out=alias)
    assert result is alias
    utils.gems_assert_equal(result, ref_alias)

    with pytest.raises(RuntimeError):
        flag_gems.msort_out(
            inp,
            out=torch.empty(inp.shape, dtype=torch.float16, device=inp.device),
        )
