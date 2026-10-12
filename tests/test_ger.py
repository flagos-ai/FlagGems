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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    # One masked shape, one exactly-divisible shape and one zero-sized shape:
    # that covers the NO_MASK branch, its masked counterpart and the early
    # return, which is every host-side decision the wrapper makes.
    GER_SHAPES = [(1, 1), (64, 64), (0, 4)]
else:
    # Both tile-geometry branches of _forward_config are selected by M * N, so
    # the list spans sizes below and above its 1 << 22 cutoff. Lopsided and
    # length-1 pairs check the degenerate grids, the odd pairs check masking,
    # and the zero-sized pairs check the early return before any launch.
    GER_SHAPES = [
        (0, 0),
        (0, 4),
        (4, 0),
        (1, 1),
        (1, 256),
        (256, 1),
        (32, 64),
        (64, 32),
        (64, 64),
        (97, 33),
        (256, 256),
        (512, 1024),
        (1024, 512),
        (1024, 1024),
        (4096, 1024),
    ]


def _rand(shape, dtype, requires_grad=False):
    if dtype == torch.bool:
        return torch.randint(0, 2, (shape,), device=flag_gems.device).to(torch.bool)
    if not dtype.is_floating_point and not dtype.is_complex:
        hi = 8 if dtype == torch.int8 else 128
        lo = 0 if dtype == torch.uint8 else -hi
        return torch.randint(lo, hi, (shape,), dtype=dtype, device=flag_gems.device)
    inp = torch.randn((shape,), dtype=dtype, device=flag_gems.device)
    return inp.requires_grad_() if requires_grad else inp


@pytest.mark.ger
@pytest.mark.parametrize("M, N", GER_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ger(M, N, dtype):
    inp = _rand(M, dtype)
    vec2 = _rand(N, dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_vec2 = utils.to_reference(vec2, True)

    ref_out = torch.ger(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ger
@pytest.mark.parametrize("M, N", GER_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.ALL_INT_DTYPES + utils.BOOL_TYPES + [torch.int8, torch.uint8]
)
def test_ger_int_bool(M, N, dtype):
    inp = _rand(M, dtype)
    vec2 = _rand(N, dtype)
    ref_inp = utils.to_reference(inp)
    ref_vec2 = utils.to_reference(vec2)

    ref_out = torch.ger(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.ger
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_ger_complex(dtype):
    if dtype != torch.complex64:
        pytest.skip("the kernel views complex operands as pairs of float32")
    inp = torch.randn((37,), dtype=dtype, device=flag_gems.device)
    vec2 = torch.randn((53,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_vec2 = utils.to_reference(vec2, True)

    ref_out = torch.ger(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.ger
def test_ger_element_order():
    # A product of two disjoint ramps makes every out[i, j] unique, so a
    # transposed or otherwise permuted write would not compare equal.
    inp = torch.arange(1, 24, dtype=torch.int32, device=flag_gems.device)
    vec2 = torch.arange(100, 117, dtype=torch.int32, device=flag_gems.device)
    ref_out = torch.ger(utils.to_reference(inp), utils.to_reference(vec2))
    utils.gems_assert_equal(flag_gems.ger(inp, vec2), ref_out)


@pytest.mark.ger
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ger_strided_input(dtype):
    # Both operands are passed to the kernel with their own stride, so a
    # non-unit-stride slice must be read through that stride rather than
    # contiguously.
    base_a = torch.randn((128,), dtype=dtype, device=flag_gems.device)
    base_b = torch.randn((128,), dtype=dtype, device=flag_gems.device)
    inp = base_a[1:17:5]
    vec2 = base_b[2:34:2]
    assert not inp.is_contiguous()
    ref_out = torch.ger(utils.to_reference(inp, True), utils.to_reference(vec2, True))
    utils.gems_assert_close(flag_gems.ger(inp, vec2), ref_out, dtype)


@pytest.mark.ger
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ger_special_values(dtype):
    inp = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.5],
        dtype=dtype,
        device=flag_gems.device,
    )
    vec2 = torch.tensor(
        [float("inf"), 0.0, float("nan"), -2.5, float("-inf"), 1.0],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_out = torch.ger(utils.to_reference(inp), utils.to_reference(vec2))
    utils.gems_assert_equal(flag_gems.ger(inp, vec2), ref_out, equal_nan=True)


@pytest.mark.ger
@pytest.mark.parametrize(
    "dtype_pair",
    [
        (torch.float16, torch.float32),
        (torch.float16, torch.bfloat16),
        (torch.int32, torch.int64),
        (torch.uint8, torch.int8),
        (torch.bool, torch.float32),
        (torch.int64, torch.float16),
    ],
)
def test_ger_mixed_dtype(dtype_pair):
    # aten::ger promotes its operands with result_type, so the output dtype has
    # to follow the promotion and not either input.
    da, db = dtype_pair
    inp = _rand(24, da)
    vec2 = _rand(18, db)
    expected = torch.result_type(inp, vec2)
    ref_out = torch.ger(utils.to_reference(inp), utils.to_reference(vec2))
    res_out = flag_gems.ger(inp, vec2)
    assert res_out.dtype == expected
    utils.gems_assert_close(res_out, ref_out, expected)


@pytest.mark.ger
@pytest.mark.parametrize("M, N", [(1, 1), (64, 64), (97, 33)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ger_out(M, N, dtype):
    # aten::ger.out writes through the caller's buffer and returns that same
    # tensor, so identity matters as much as the values.
    inp = _rand(M, dtype)
    vec2 = _rand(N, dtype)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref_out = torch.ger(utils.to_reference(inp, True), utils.to_reference(vec2, True))

    res_out = flag_gems.ger(inp, vec2, out=out)
    assert res_out is out
    utils.gems_assert_close(res_out, ref_out, dtype)

    positional = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    assert flag_gems.ger(inp, vec2, positional) is positional
    utils.gems_assert_close(positional, ref_out, dtype)


@pytest.mark.ger
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ger_backward(dtype):
    M, N = 64, 48
    inp = _rand(M, dtype, requires_grad=True)
    vec2 = _rand(N, dtype, requires_grad=True)
    ref_inp = utils.to_reference(inp, True).detach().requires_grad_()
    ref_vec2 = utils.to_reference(vec2, True).detach().requires_grad_()

    ref_out = torch.ger(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    ref_da, ref_db = torch.autograd.grad(ref_out, (ref_inp, ref_vec2), ref_grad)
    res_da, res_db = torch.autograd.grad(res_out, (inp, vec2), out_grad)

    utils.gems_assert_close(res_da, ref_da, dtype, reduce_dim=N)
    utils.gems_assert_close(res_db, ref_db, dtype, reduce_dim=M)


@pytest.mark.ger
@pytest.mark.parametrize(
    "inp_shape, vec2_shape", [((), (4,)), ((4,), ()), ((2, 3), (4,)), ((4,), (2, 3))]
)
def test_ger_rejects_invalid_rank(inp_shape, vec2_shape):
    inp = torch.randn(inp_shape, device=flag_gems.device)
    vec2 = torch.randn(vec2_shape, device=flag_gems.device)
    with pytest.raises(ValueError):
        flag_gems.ger(inp, vec2)


@pytest.mark.ger
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_ger_rejects_unsupported_dtype(dtype):
    inp = torch.randn((8,), device=flag_gems.device).to(dtype)
    vec2 = torch.randn((8,), device=flag_gems.device).to(dtype)
    with pytest.raises(RuntimeError):
        flag_gems.ger(inp, vec2)


@pytest.mark.ger
def test_ger_rejects_bad_out():
    inp = torch.randn((8,), device=flag_gems.device)
    vec2 = torch.randn((6,), device=flag_gems.device)
    with pytest.raises(ValueError):
        flag_gems.ger(inp, vec2, out=torch.empty((8, 5), device=flag_gems.device))
    with pytest.raises(ValueError):
        flag_gems.ger(
            inp,
            vec2,
            out=torch.empty((8, 6), dtype=torch.float16, device=flag_gems.device),
        )
    with pytest.raises(TypeError):
        flag_gems.ger(inp, vec2, out=object())


@pytest.mark.ger
def test_ger_rejects_non_tensor():
    inp = torch.randn((8,), device=flag_gems.device)
    with pytest.raises(TypeError):
        flag_gems.ger(inp, 3.0)
    with pytest.raises(TypeError):
        flag_gems.ger(3.0, inp)
