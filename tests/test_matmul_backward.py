import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# (self_shape, other_shape) pairs covering the matmul broadcasting rules:
# 2D x 2D, batched (bmm), 1D vectors, and broadcasted batch dims.
MATMUL_SHAPES = [
    ((16, 32), (32, 24)),
    ((64, 128), (128, 64)),
    ((4, 16, 32), (4, 32, 24)),
    ((8, 32, 64), (8, 64, 16)),
    ((2, 3, 16, 32), (2, 3, 32, 24)),
    ((32,), (32,)),  # dot -> scalar
    ((32,), (32, 24)),  # vec @ mat
    ((16, 32), (32,)),  # mat @ vec
    ((4, 16, 32), (32, 24)),  # batched @ 2D
    ((16, 32), (4, 32, 24)),  # 2D @ batched
    ((5, 1, 16, 32), (1, 3, 32, 24)),  # broadcast batch dims
]


def _reference_grads(self_t, other_t, grad_t):
    self_t = self_t.detach().requires_grad_(True)
    other_t = other_t.detach().requires_grad_(True)
    out = torch.matmul(self_t, other_t)
    gs, go = torch.autograd.grad(out, [self_t, other_t], grad_t)
    return gs, go


@pytest.mark.matmul_backward
@pytest.mark.parametrize("self_shape, other_shape", MATMUL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matmul_backward(self_shape, other_shape, dtype):
    self_t = torch.randn(self_shape, dtype=dtype, device=flag_gems.device)
    other_t = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    out = torch.matmul(self_t, other_t)
    grad = torch.randn_like(out)

    ref_self = utils.to_reference(self_t, True)
    ref_other = utils.to_reference(other_t, True)
    ref_grad = utils.to_reference(grad, True)
    ref_gs, ref_go = _reference_grads(ref_self, ref_other, ref_grad)

    with flag_gems.use_gems():
        res_gs, res_go = torch.ops.aten.matmul_backward(
            grad, self_t, other_t, [True, True]
        )

    utils.gems_assert_close(res_gs, ref_gs, dtype, reduce_dim=self_t.shape[-1])
    utils.gems_assert_close(res_go, ref_go, dtype, reduce_dim=self_t.shape[-1])


@pytest.mark.matmul_backward
@pytest.mark.parametrize("self_shape, other_shape", MATMUL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matmul_backward_self_only(self_shape, other_shape, dtype):
    """output_mask = (True, False): only grad_self is computed."""
    self_t = torch.randn(self_shape, dtype=dtype, device=flag_gems.device)
    other_t = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    out = torch.matmul(self_t, other_t)
    grad = torch.randn_like(out)

    ref_self = utils.to_reference(self_t, True)
    ref_other = utils.to_reference(other_t, True)
    ref_grad = utils.to_reference(grad, True)
    ref_gs, _ = _reference_grads(ref_self, ref_other, ref_grad)

    with flag_gems.use_gems():
        res_gs, res_go = torch.ops.aten.matmul_backward(
            grad, self_t, other_t, [True, False]
        )

    assert res_go is None
    utils.gems_assert_close(res_gs, ref_gs, dtype, reduce_dim=self_t.shape[-1])


@pytest.mark.matmul_backward
@pytest.mark.parametrize("self_shape, other_shape", MATMUL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_matmul_backward_other_only(self_shape, other_shape, dtype):
    """output_mask = (False, True): only grad_other is computed."""
    self_t = torch.randn(self_shape, dtype=dtype, device=flag_gems.device)
    other_t = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    out = torch.matmul(self_t, other_t)
    grad = torch.randn_like(out)

    ref_self = utils.to_reference(self_t, True)
    ref_other = utils.to_reference(other_t, True)
    ref_grad = utils.to_reference(grad, True)
    _, ref_go = _reference_grads(ref_self, ref_other, ref_grad)

    with flag_gems.use_gems():
        res_gs, res_go = torch.ops.aten.matmul_backward(
            grad, self_t, other_t, [False, True]
        )

    assert res_gs is None
    utils.gems_assert_close(res_go, ref_go, dtype, reduce_dim=self_t.shape[-1])
