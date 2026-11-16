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

GRAD_SIDES = ["both", "left", "right"]
SCALARS = [0, 0.25]


@pytest.mark.lt_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lt_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)
    ref_out = ref_inp1.lt_(ref_inp2)

    res_out = flag_gems.lt_(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out is inp1


@pytest.mark.lt_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lt_scalar_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar = 0

    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = ref_inp.lt_(scalar)

    res_out = flag_gems.lt_scalar_(inp, scalar)

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out is inp


def _working_pair(inp1, inp2, left_grad, right_grad, reference=False):
    """Build the leaves and the non-leaf working copies for one comparison.

    ``lt_`` is in place, so ATen refuses to write into a leaf that requires a
    gradient.  Both the reference and the candidate therefore mutate non-leaf
    working copies and differentiate with respect to the leaves, which is the
    autograd path a training graph uses.

    ``reference=True`` builds the native side through ``to_reference`` so that
    ``--ref cpu`` runs the PyTorch baseline on the CPU; the candidate side keeps
    the tensors on the device FlagGems operates on.
    """
    lhs = utils.to_reference(inp1, True) if reference else inp1
    rhs = utils.to_reference(inp2, True) if reference else inp2
    lhs = lhs.detach().clone()
    rhs = rhs.detach().clone()
    lhs.requires_grad_(left_grad)
    rhs.requires_grad_(right_grad)
    return lhs, lhs.clone(), rhs, rhs.clone()


def _grad_inputs(leaves, out, dout):
    """Differentiate ``out`` with respect to the leaves that need a gradient."""
    needed = [leaf for leaf in leaves if leaf.requires_grad]
    grad_output = dout.to(device=out.device, dtype=out.dtype)
    return list(torch.autograd.grad(out, needed, grad_output))


@pytest.mark.lt_backward_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("grad_side", GRAD_SIDES)
def test_lt_tensor_backward(shape, dtype, grad_side):
    """Compare the ``lt_.Tensor`` gradients against native PyTorch autograd."""
    left_grad = grad_side in ("both", "left")
    right_grad = grad_side in ("both", "right")

    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_lhs, ref_work_lhs, ref_rhs, ref_work_rhs = _working_pair(
        inp1, inp2, left_grad, right_grad, reference=True
    )
    ref_out = ref_work_lhs.lt_(ref_work_rhs)

    lhs, work_lhs, rhs, work_rhs = _working_pair(inp1, inp2, left_grad, right_grad)
    res_out = flag_gems.lt_(work_lhs, work_rhs)

    # The in-place contract: same object, same dtype, same values and a single
    # version bump of the receiver.
    assert res_out is work_lhs
    assert res_out.dtype == dtype
    assert work_lhs._version == 1
    utils.gems_assert_close(res_out, ref_out, dtype)

    # The comparison is not differentiable, so every differentiable input gets
    # an exactly zero gradient while an unused side stays absent.
    dout = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_grads = _grad_inputs((ref_lhs, ref_rhs), ref_out, dout)
    res_grads = _grad_inputs((lhs, rhs), res_out, dout)
    assert len(res_grads) == len(ref_grads)
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        assert res_grad.shape == ref_grad.shape
        utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.lt_backward_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("scalar", SCALARS)
def test_lt_scalar_backward(shape, dtype, scalar):
    """Compare the ``lt_.Scalar`` gradient against native PyTorch autograd."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_leaf = utils.to_reference(inp, True).detach().clone().requires_grad_(True)
    ref_out = ref_leaf.clone().lt_(scalar)

    res_leaf = inp.detach().clone().requires_grad_(True)
    res_out = flag_gems.lt_scalar_(res_leaf.clone(), scalar)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype)

    dout = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    (ref_grad,) = _grad_inputs((ref_leaf,), ref_out, dout)
    (res_grad,) = _grad_inputs((res_leaf,), res_out, dout)
    assert res_grad.shape == ref_grad.shape
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.lt_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lt_backward_broadcast(dtype):
    """Broadcast operands keep zero gradients with their own shapes."""
    lhs_shape, rhs_shape = (4, 3), (4, 1)
    inp1 = torch.randn(lhs_shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(rhs_shape, dtype=dtype, device=flag_gems.device)

    ref_lhs, ref_work_lhs, ref_rhs, ref_work_rhs = _working_pair(
        inp1, inp2, True, True, reference=True
    )
    ref_out = ref_work_lhs.lt_(ref_work_rhs)

    lhs, work_lhs, rhs, work_rhs = _working_pair(inp1, inp2, True, True)
    res_out = flag_gems.lt_(work_lhs, work_rhs)

    assert res_out is work_lhs
    assert tuple(res_out.shape) == lhs_shape
    utils.gems_assert_close(res_out, ref_out, dtype)

    dout = torch.randn(lhs_shape, dtype=dtype, device=flag_gems.device)
    ref_grads = _grad_inputs((ref_lhs, ref_rhs), ref_out, dout)
    res_grads = _grad_inputs((lhs, rhs), res_out, dout)
    assert [tuple(g.shape) for g in res_grads] == [lhs_shape, rhs_shape]
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.lt_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lt_backward_leaf_rejected(dtype):
    """A leaf that requires grad cannot be used as the in-place receiver."""
    leaf = torch.randn(16, dtype=dtype, device=flag_gems.device)
    leaf.requires_grad_(True)
    other = torch.randn(16, dtype=dtype, device=flag_gems.device)
    snapshot = leaf.detach().clone()

    with pytest.raises(RuntimeError, match="in-place operation"):
        flag_gems.lt_(leaf, other)
    # The guard fires before the kernel writes, so the leaf is untouched.
    assert torch.equal(leaf.detach(), snapshot)

    with pytest.raises(RuntimeError, match="in-place operation"):
        flag_gems.lt_scalar_(leaf, 0)
    assert torch.equal(leaf.detach(), snapshot)


@pytest.mark.lt_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lt_backward_through_dispatch(dtype):
    """The registered ``torch.lt_`` entry must reach the FlagGems backward."""
    inp = torch.randn(32, dtype=dtype, device=flag_gems.device)
    other = torch.randn(32, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        leaf = inp.detach().clone().requires_grad_(True)
        out = leaf.clone().lt_(other)
        (grad,) = torch.autograd.grad(out, (leaf,), torch.ones_like(out))

    assert type(out.grad_fn).__name__ == "_LtInplaceFunctionBackward"
    assert grad.shape == inp.shape and grad.dtype == dtype
    assert torch.equal(grad, torch.zeros_like(grad))
