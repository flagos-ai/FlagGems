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
from . import conftest as cfg

if cfg.QUICK_MODE:
    ROUND_DECIMALS_SHAPES = [(2, 3)]
    ROUND_HALF_SHAPES = [(2, 3)]
    ROUND_DECIMALS = [0]
else:
    ROUND_DECIMALS_SHAPES = [(2, 3), (128, 256), (4, 8, 16)]
    ROUND_HALF_SHAPES = [(2, 3), (4, 8)]
    ROUND_DECIMALS = [-2, -1, 0, 1, 2]


@pytest.mark.round
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.round(ref_inp)
    res_out = flag_gems.round(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.round_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.round_(ref_inp)
    res_out = flag_gems.round_(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.round_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.round(ref_inp, out=ref_out)
    flag_gems.round_out(inp, out=out)

    utils.gems_assert_equal(out, ref_out)


@pytest.mark.round
@pytest.mark.parametrize("shape", ROUND_DECIMALS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("decimals", ROUND_DECIMALS)
def test_round_decimals(shape, dtype, decimals):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 100

    # When decimals!=0 the two implementations can disagree by one quantisation
    # step (10^-decimals). FlagGems rounds the exact quotient; ATen scales the
    # input by a power of ten in the input dtype first, so a value sitting just
    # below a .5 boundary (e.g. -249.014999 at decimals=2) can round the other
    # way. Low-precision dtypes add an ULP of the result magnitude on top.
    ref_inp = utils.to_reference(inp)
    ref_out = torch.round(ref_inp, decimals=decimals).to("cpu")

    if decimals != 0:
        ulp = float(torch.finfo(dtype).eps * inp.abs().max().float().clamp(min=1.0))
        atol = max(10.0 ** (-decimals), ulp)
    else:
        atol = 0

    res_out = flag_gems.round(inp, decimals=decimals).to("cpu")

    if atol:
        utils.gems_assert_close(
            res_out.float(), ref_out.float(), torch.float32, atol=atol
        )
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.round
@pytest.mark.parametrize("shape", ROUND_HALF_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_round_half_to_even(shape, dtype):
    # Test round half to even: 2.5->2, 3.5->4, -2.5->-2, -3.5->-4
    inp = torch.tensor(
        [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.round(ref_inp)
    res_out = flag_gems.round(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.round
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_round_transposed_bug90(dtype, monkeypatch):
    torch.manual_seed(0)
    base_cpu = torch.randn(128, 128, dtype=torch.float32).to(dtype)
    base_dev = base_cpu.to(flag_gems.device)
    base_before = base_dev.clone()

    inp = base_dev.transpose(0, 1)
    ref_inp = utils.to_reference(base_dev).transpose(0, 1)
    assert not inp.is_contiguous()
    assert inp.stride() == (1, 128)

    ref_out = torch.round(ref_inp)
    res_direct = flag_gems.round(inp)
    utils.gems_assert_equal(res_direct, ref_out)
    assert res_direct.data_ptr() != inp.data_ptr()

    with monkeypatch.context() as registration:
        registration.setattr(flag_gems, "current_work_registrar", None)
        library = torch.library.Library("aten", "IMPL")
        try:
            flag_gems.only_enable(lib=library, include=["round"])
            res_dispatch = torch.round(inp)
        finally:
            library._destroy()
    utils.gems_assert_equal(res_dispatch, ref_out)
    utils.gems_assert_equal(base_dev, utils.to_reference(base_before))


@pytest.mark.round
@pytest.mark.parametrize(
    "base_shape,make_view",
    [
        ((33,), lambda t: t[3::2]),
        ((17, 29), lambda t: t.transpose(0, 1)),
        ((24, 30), lambda t: t[1::2, 2::3].transpose(0, 1)),
        ((4, 8, 16), lambda t: t.permute(2, 0, 1)),
        ((5, 7, 9, 11), lambda t: t[1:, ::2, 1::2, :].permute(3, 1, 2, 0)),
        ((0, 16), lambda t: t.transpose(0, 1)),
    ],
)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_round_noncontiguous_views(base_shape, make_view, dtype):
    base_dev = torch.randn(base_shape, dtype=dtype, device=flag_gems.device) * 10
    ref_base = utils.to_reference(base_dev)

    inp = make_view(base_dev)
    ref_inp = make_view(ref_base)

    ref_out = torch.round(ref_inp)
    res_out = flag_gems.round(inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.round
@pytest.mark.round_
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_round_inplace_noncontiguous(dtype):
    base_dev = torch.randn((20, 26), dtype=dtype, device=flag_gems.device) * 10
    ref_base = utils.to_reference(base_dev.clone())

    view_dev = base_dev[1::2, 2::3].transpose(0, 1)
    view_ref = ref_base[1::2, 2::3].transpose(0, 1)
    orig_ptr = view_dev.data_ptr()
    orig_stride = view_dev.stride()

    torch.round_(view_ref)
    res = flag_gems.round_(view_dev)

    assert res is view_dev
    assert res.data_ptr() == orig_ptr
    assert res.stride() == orig_stride
    utils.gems_assert_equal(view_dev, view_ref)
    utils.gems_assert_equal(base_dev, ref_base)


@pytest.mark.round
@pytest.mark.round_out
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_round_out_noncontiguous(dtype):
    inp_base = torch.randn((18, 24), dtype=dtype, device=flag_gems.device) * 10
    ref_inp_base = utils.to_reference(inp_base)
    inp = inp_base[1::2, ::2].transpose(0, 1)
    ref_inp = ref_inp_base[1::2, ::2].transpose(0, 1)

    # Non-contiguous input -> contiguous out
    out_contig = torch.empty(inp.shape, dtype=dtype, device=flag_gems.device)
    ref_out_contig = torch.empty(ref_inp.shape, dtype=dtype, device=ref_inp.device)
    torch.round(ref_inp, out=ref_out_contig)
    res_contig = flag_gems.round_out(inp, out=out_contig)
    assert res_contig is out_contig
    utils.gems_assert_equal(out_contig, ref_out_contig)

    # Non-contiguous input -> non-contiguous out (verifies untouched neighbors)
    out_base = torch.full((24, 27), -77.0, dtype=dtype, device=flag_gems.device)
    ref_out_base = utils.to_reference(out_base.clone())
    out_view = out_base[::2, 1::3]
    ref_out_view = ref_out_base[::2, 1::3]
    torch.round(ref_inp, out=ref_out_view)
    res_view = flag_gems.round_out(inp, out=out_view)
    assert res_view is out_view
    utils.gems_assert_equal(out_view, ref_out_view)
    utils.gems_assert_equal(out_base, ref_out_base)


@pytest.mark.round
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("decimals", ROUND_DECIMALS)
def test_round_decimals_noncontiguous(dtype, decimals):
    base_dev = torch.randn((16, 24), dtype=dtype, device=flag_gems.device) * 100
    ref_base = utils.to_reference(base_dev.clone())

    inp = base_dev[1::2, ::3].transpose(0, 1)
    ref_inp = ref_base[1::2, ::3].transpose(0, 1)

    ref_out = torch.round(ref_inp, decimals=decimals).to("cpu")
    res_out = flag_gems.round(inp, decimals=decimals).to("cpu")

    if decimals != 0:
        ulp = float(torch.finfo(dtype).eps * inp.abs().max().float().clamp(min=1.0))
        atol = max(10.0 ** (-decimals), ulp)
        utils.gems_assert_close(
            res_out.float(), ref_out.float(), torch.float32, atol=atol
        )
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.round
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_round_ties_and_special_values_noncontiguous(dtype):
    vals = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        3.5,
        -3.5,
        -0.2,
        float("inf"),
        float("-inf"),
        float("nan"),
        64.5,
        -64.5,
    ]
    base_cpu = torch.tensor(vals, dtype=dtype).reshape(4, 4)
    base_dev = base_cpu.to(flag_gems.device)

    inp = base_dev.transpose(0, 1)
    ref_inp = utils.to_reference(base_dev).transpose(0, 1)

    ref_out = torch.round(ref_inp).to("cpu")
    res_out = flag_gems.round(inp).to("cpu")

    torch.testing.assert_close(res_out, ref_out, rtol=0, atol=0, equal_nan=True)
    finite_mask = ~torch.isnan(ref_out)
    assert torch.equal(
        torch.signbit(res_out[finite_mask]), torch.signbit(ref_out[finite_mask])
    )


@utils._SKIP_FP64
@pytest.mark.round
def test_round_fp64_near_tie_noncontiguous():
    base_cpu = torch.tensor(
        [[1.49999999995, -1.49999999995], [2.5, -2.5]], dtype=torch.float64
    )
    inp = base_cpu.to(flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(base_cpu.to(flag_gems.device)).transpose(0, 1)

    utils.gems_assert_equal(flag_gems.round(inp), torch.round(ref_inp))
