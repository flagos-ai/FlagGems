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


@pytest.mark.arcsin
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.arcsin(ref_inp)
    res_out = flag_gems.arcsin(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, True)


@pytest.mark.arcsin_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.arcsin(ref_inp)
    res_out = flag_gems.arcsin_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, True)
    utils.gems_assert_close(inp, ref_out, dtype, True)


@pytest.mark.arcsin_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.empty_like(ref_inp)
    torch.arcsin(ref_inp, out=ref_out)
    res_out = torch.empty_like(inp)
    flag_gems.arcsin(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, True)


@pytest.mark.arcsin_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES + utils.BOOL_TYPES)
def test_arcsin_out_integer_rejects(shape, dtype):
    # ATen raises for an integer/bool out buffer (the float result cannot be
    # cast into it); FlagGems mirrors that instead of silently truncating.
    inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.arcsin_out(inp, out=torch.empty_like(inp))


@pytest.mark.arcsin
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_boundaries_and_out_of_domain(dtype):
    values = torch.tensor(
        [-1.0, -0.5, 0.0, 0.5, 1.0, -1.5, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_values = utils.to_reference(values)

    ref_out = torch.arcsin(ref_values)
    res_out = flag_gems.arcsin(values)

    utils.gems_assert_close(res_out, ref_out, dtype, True)
    assert torch.isnan(res_out[-2:]).all()


@pytest.mark.arcsin
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_empty_and_scalar(dtype):
    empty = torch.empty((0,), dtype=dtype, device=flag_gems.device)
    scalar = torch.tensor(0.5, dtype=dtype, device=flag_gems.device)

    ref_empty = utils.to_reference(empty)
    ref_scalar = utils.to_reference(scalar)

    res_empty = flag_gems.arcsin(empty)
    res_scalar = flag_gems.arcsin(scalar)

    utils.gems_assert_close(res_empty, torch.arcsin(ref_empty), dtype, True)
    utils.gems_assert_close(res_scalar, torch.arcsin(ref_scalar), dtype, True)


# ---------------------------------------------------------------------------
# Domain-matrix additions: value-range coverage ([-1,1], [0,1], [-1,0] and the
# out-of-domain edges beyond both bounds) crossed with shapes and dtypes.
# ---------------------------------------------------------------------------

ARCSIN_VALUE_RANGES = ["full_domain", "positive", "negative", "above_max", "below_min"]


def _arcsin_range_input(shape, dtype, range_name):
    """Build inputs for the arcsin domain matrix: valid domain [-1, 1] plus the
    out-of-domain bands beyond both endpoints (ATen yields NaN there)."""
    if range_name == "full_domain":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -1.0, 1.0
        )
    if range_name == "positive":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            0.0, 1.0
        )
    if range_name == "negative":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -1.0, 0.0
        )
    if range_name == "above_max":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            1.5, 4.0
        )
    if range_name == "below_min":
        return torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(
            -4.0, -1.5
        )
    raise ValueError(f"Unknown range_name: {range_name}")


@pytest.mark.arcsin
@pytest.mark.parametrize("range_name", ARCSIN_VALUE_RANGES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_range(shape, dtype, range_name):
    inp = _arcsin_range_input(shape, dtype, range_name)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.arcsin(ref_inp)
    res_out = flag_gems.arcsin(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, True)


@pytest.mark.arcsin
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_special_values(dtype):
    # Param coverage type 5: signed zero maps to itself; nan stays nan and the
    # infinities are out of domain.
    values = torch.tensor(
        [0.0, -0.0, float("nan"), float("inf"), -float("inf")],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_values = utils.to_reference(values, True)

    ref_out = torch.arcsin(ref_values)
    res_out = flag_gems.arcsin(values)

    utils.gems_assert_close(res_out, ref_out, dtype, True)
    # Only the zero lanes carry a meaningful sign bit; NaN sign bits are
    # platform-defined, so they are not compared.
    assert torch.equal(
        torch.signbit(res_out[:2].cpu()), torch.signbit(ref_out[:2].cpu())
    )


@pytest.mark.arcsin
@pytest.mark.parametrize("shape", utils.POINTWISE_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_noncontiguous(shape, dtype):
    # Param coverage type 5: non-contiguous inputs must produce the same
    # values as their contiguous copies. Shapes with any 1-sized dim give
    # a contiguous result (stride-1 squeeze), and 0-dim has no stride at
    # all, so both are excluded from the shape list.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noncontig = inp.repeat(2, *[1] * (inp.ndim - 1))[::2]
    assert not noncontig.is_contiguous()
    ref_inp = utils.to_reference(noncontig.contiguous(), True)

    ref_out = torch.arcsin(ref_inp)
    res_out = flag_gems.arcsin(noncontig)

    utils.gems_assert_close(res_out, ref_out, dtype, True)


@pytest.mark.arcsin
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_zero_size(dtype):
    # Param coverage type 5: 0-size tensors must pass through unchanged.
    for shape in [(0,), (0, 3), (2, 0, 4)]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)

        ref_out = torch.arcsin(ref_inp)
        res_out = flag_gems.arcsin(inp)

        utils.gems_assert_close(res_out, ref_out, dtype, True)
        assert res_out.numel() == 0


@pytest.mark.arcsin_
@pytest.mark.parametrize("range_name", ARCSIN_VALUE_RANGES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin__range(shape, dtype, range_name):
    inp = _arcsin_range_input(shape, dtype, range_name)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = torch.arcsin_(ref_inp)
    res_out = flag_gems.arcsin_(inp)

    ref_out = ref_out.to(res_out.dtype)
    ref_inp = ref_inp.to(inp.dtype)
    utils.gems_assert_close(res_out, ref_out, dtype, True)
    utils.gems_assert_close(inp, ref_inp, dtype, True)


@pytest.mark.arcsin_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin__inplace_identity(shape, dtype):
    # In-place contract: the returned tensor is the input object itself.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.arcsin_(inp)
    assert res_out is inp


@pytest.mark.arcsin_
@pytest.mark.parametrize("shape", utils.POINTWISE_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin__noncontiguous(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noncontig = inp.repeat(2, *[1] * (inp.ndim - 1))[::2]
    assert not noncontig.is_contiguous()
    ref_inp = utils.to_reference(noncontig.contiguous().clone(), True)

    ref_out = torch.arcsin_(ref_inp)
    res_out = flag_gems.arcsin_(noncontig)

    ref_out = ref_out.to(res_out.dtype)
    utils.gems_assert_close(res_out, ref_out, dtype, True)


@pytest.mark.arcsin_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin__zero_size(dtype):
    for shape in [(0,), (0, 3), (2, 0, 4)]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        res_out = flag_gems.arcsin_(inp)
        assert res_out.numel() == 0
        assert res_out.dtype == dtype


@pytest.mark.arcsin_out
@pytest.mark.parametrize("range_name", ARCSIN_VALUE_RANGES)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_out_range(shape, dtype, range_name):
    inp = _arcsin_range_input(shape, dtype, range_name)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.empty_like(ref_inp)
    torch.arcsin(ref_inp, out=ref_out)
    res_out = torch.empty_like(inp)
    flag_gems.arcsin(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, True)


@pytest.mark.arcsin_out
@pytest.mark.parametrize("shape", utils.POINTWISE_NONCONTIG_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_out_noncontiguous(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    noncontig = inp.repeat(2, *[1] * (inp.ndim - 1))[::2]
    assert not noncontig.is_contiguous()
    ref_inp = utils.to_reference(noncontig.contiguous(), True)

    ref_out = torch.empty_like(ref_inp)
    torch.arcsin(ref_inp, out=ref_out)
    res_out = torch.empty_like(noncontig)
    flag_gems.arcsin(noncontig, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype, True)


@pytest.mark.arcsin_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_arcsin_out_zero_size(dtype):
    for shape in [(0,), (0, 3), (2, 0, 4)]:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        res_out = torch.empty_like(inp)
        flag_gems.arcsin(inp, out=res_out)
        assert res_out.numel() == 0
