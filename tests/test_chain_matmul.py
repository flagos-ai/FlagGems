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
from . import test_utils as tu

# Multiply compatible rank-2 matrices; keep native-dtype intermediate rounding and overflow.
_CHAIN_SHAPES = tu.selected_cases(
    [
        [(1, 1)],
        [(4, 8)],
        [(2, 3), (3, 4)],
        [(1, 5), (5, 1), (1, 7)],
        [(16, 32), (32, 64), (64, 32), (32, 16)],
        [(8, 16), (16, 32), (32, 48), (48, 32), (32, 16)],
        [(33, 65), (65, 17), (17, 129), (129, 255), (255, 71)],
    ],
    quick=[[(2, 3), (3, 4)]],
)

_OUT_CHAIN_SHAPES = _CHAIN_SHAPES[:4]

_BACKWARD_CHAINS = tu.selected_cases(
    [
        [(4, 8)],
        [(2, 3), (3, 4)],
        [(4, 8), (8, 16), (16, 4)],
        [(16, 32), (32, 64), (64, 32), (32, 16)],
    ],
    quick=[[(2, 3), (3, 4)]],
)

_NONCONTIG_CHAINS = tu.selected_cases(
    [[(4, 8), (8, 16)], [(16, 32), (32, 64), (64, 16)]], quick=[[(4, 8), (8, 16)]]
)


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", _CHAIN_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_chain_matmul(shapes, value_range, dtype):
    inp = [tu.make_input(dtype, shape, value_range) for shape in shapes]
    ref_inp = [tu.to_reference(m) for m in inp]

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    res_out = flag_gems.chain_matmul(inp)

    assert res_out.is_contiguous()
    tu.assert_result_close(res_out, ref_out.to(dtype))


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", _NONCONTIG_CHAINS)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_chain_matmul_non_contiguous(shapes, dtype):
    inp = [tu.make_input(dtype, (cols, rows), ["-1", "1"]).t() for rows, cols in shapes]
    assert all(not m.is_contiguous() for m in inp)
    ref_inp = [tu.to_reference(m) for m in inp]

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    res_out = flag_gems.chain_matmul(inp)

    tu.assert_result_close(res_out, ref_out.to(dtype))


@pytest.mark.chain_matmul_out
@pytest.mark.parametrize("shapes", _OUT_CHAIN_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_chain_matmul_out(shapes, value_range, dtype):
    inp = [tu.make_input(dtype, shape, value_range) for shape in shapes]
    ref_inp = [tu.to_reference(m) for m in inp]

    out_shape = (shapes[0][0], shapes[-1][1])
    # Garbage-prefilled buffers: the .out overload must overwrite every element.
    out = torch.full(out_shape, 7.0, dtype=dtype, device=flag_gems.device)
    ref_out = torch.full(
        out_shape, 7.0, dtype=ref_inp[0].dtype, device=ref_inp[0].device
    )

    torch.ops.aten.chain_matmul.out(ref_inp, out=ref_out)
    res_ret = flag_gems.chain_matmul(inp, out=out)

    # The .out overload must write into and return the caller's buffer.
    assert res_ret is out
    tu.assert_result_close(out, ref_out.to(dtype))


@pytest.mark.chain_matmul
@pytest.mark.parametrize(
    "dtype,scenario", tu.selected_cases(tu.special_value_cases(utils.ALL_FLOAT_DTYPES))
)
def test_chain_matmul_nan_inf(dtype, scenario):
    # Zeros in m2 also exercise Inf * 0 -> NaN during reduction.
    m1 = tu.make_special_input(dtype, scenario)[:4].reshape(2, 2)
    m2 = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = [m1, m2]
    ref_inp = [tu.to_reference(m) for m in inp]

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    res_out = flag_gems.chain_matmul(inp)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", _BACKWARD_CHAINS)
@pytest.mark.parametrize("dtype", tu.selected_cases(utils.ALL_FLOAT_DTYPES))
def test_chain_matmul_backward(shapes, dtype):
    inp = [
        tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_() for shape in shapes
    ]
    grad = tu.make_input(dtype, (shapes[0][0], shapes[-1][1]), ["-1", "1"])

    ref_inp = []
    for matrix in inp:
        ref_matrix = tu.to_reference(matrix.detach())
        ref_inp.append(ref_matrix.requires_grad_())
    ref_grad = tu.to_reference(grad)

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    ref_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    # The candidate forward must match the native-dtype reference...
    res_out = flag_gems.chain_matmul(inp)
    tu.assert_result_close(res_out, ref_out.to(dtype))

    assert res_out.requires_grad
    res_grads = torch.autograd.grad(res_out, inp, grad_outputs=grad)
    for res_g, ref_g in zip(res_grads, ref_grads):
        assert res_g.dtype == dtype
        assert res_g.shape == ref_g.shape
        tu.assert_result_close(res_g, ref_g.to(dtype))


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_empty_list():
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul([])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul([])


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.chain_matmul(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(3.14)


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_1d_matrix():
    inp = [tu.make_input(torch.float32, (4,), ["-1", "1"])]
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(inp)


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_3d_tensor():
    matrix = tu.make_input(torch.float32, (2, 3, 4), ["-1", "1"])
    inp = [matrix, matrix]
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(inp)


@pytest.mark.chain_matmul_negative
def test_chain_matmul_rejects_mismatched_dims():
    inp = [
        tu.make_input(torch.float32, (3, 4), ["-1", "1"]),
        tu.make_input(torch.float32, (5, 6), ["-1", "1"]),
    ]
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(inp)


@pytest.mark.chain_matmul_negative
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="aten chain_matmul accepts integer addmm on the CPU reference path",
)
def test_chain_matmul_rejects_int_dtype():
    inp = [
        tu.make_input(torch.int32, (2, 2), ["0", "1"]),
        tu.make_input(torch.int32, (2, 2), ["0", "1"]),
    ]
    # On the accelerator the integer addmm is not implemented.
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(inp)


@pytest.mark.chain_matmul_out_negative
def test_chain_matmul_out_rejects_wrong_dtype():
    inp = [
        tu.make_input(torch.float32, shape, ["-1", "1"]) for shape in [(4, 8), (8, 4)]
    ]
    ref_inp = [tu.to_reference(m) for m in inp]

    ref_bad = torch.empty(4, 4, dtype=torch.int32, device=ref_inp[0].device)
    res_bad = torch.empty(4, 4, dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul.out(ref_inp, out=ref_bad)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(inp, out=res_bad)
