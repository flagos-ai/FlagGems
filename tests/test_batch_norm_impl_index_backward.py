import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

# Shapes representative of common batch-norm workloads (2D up to 5D).  The
# feature dimension is always ``shape[1]``.
SHAPES = [
    (16, 3),
    (32, 32, 32),
    (8, 32, 224, 224),
    (2050, 16, 32, 32),
    (8, 16, 3, 224, 224),
]


def _forward_save_stats(shape, dtype, device, weight, bias, train, impl_index):
    """Run the forward pass for the requested backend and return the saved
    statistics that the corresponding ``_batch_norm_impl_index_backward`` call
    expects, mirroring what a JIT-traced model would feed back in.

    The forward returns ``(save_mean, save_var_transform, running_mean,
    running_var, reserve)`` where ``save_var_transform`` holds the *inverse
    standard deviation* for every training backend.
    """
    inp = torch.randn(shape, dtype=dtype, device=device)
    running_mean = torch.zeros(shape[1], dtype=dtype, device=device)
    running_var = torch.ones(shape[1], dtype=dtype, device=device)

    if impl_index == 1:
        # cudnn_batch_norm requires a (float32) weight and bias even when the
        # model is not affine, so substitute neutral ones/zeros in that case.
        fwd_weight = (
            torch.ones(shape[1], dtype=dtype, device=device)
            if weight is None
            else weight
        )
        fwd_bias = (
            torch.zeros(shape[1], dtype=dtype, device=device) if bias is None else bias
        )
        out, save_mean, save_var, reserve = torch.ops.aten.cudnn_batch_norm(
            inp.to(torch.float32),
            fwd_weight.to(torch.float32),
            fwd_bias.to(torch.float32),
            running_mean.to(torch.float32) if train else None,
            running_var.to(torch.float32) if train else None,
            train,
            1e-5,
            False,
        )
        # cudnn returns the saved statistics in float32 (the accumulate type),
        # which is exactly what the backward expects, so no cast is needed.
        return inp, save_mean, save_var, running_mean, running_var, reserve
    else:
        # impl_index == 0 -> native backend.  native_batch_norm always saves
        # the inverse standard deviation.
        out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
            inp, weight, bias, running_mean, running_var, train, 0.1, 1e-5
        )
        reserve = torch.empty(0, dtype=torch.uint8, device=device)
        return inp, save_mean, save_invstd, running_mean, running_var, reserve


@pytest.mark.batch_norm_impl_index_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("impl_index", [0, 1])
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_impl_index_backward_train(shape, dtype, impl_index, affine):
    """Training-mode backward: the saved statistics hold the inverse std."""
    # The cudnn backend (impl_index == 1) has no CPU implementation of
    # ``aten::cudnn_batch_norm_backward`` (CUDA-only), so under ``--ref=cpu`` the
    # reference (which runs on the CPU reference device) cannot dispatch it.
    # Skip this combination in quick-cpu mode; the native backend
    # (impl_index == 0) and the eval-mode cudnn path both have CPU references.
    if cfg.TO_CPU and impl_index == 1:
        pytest.skip(
            "cudnn_batch_norm_backward has no CPU backend; "
            "skip train + impl_index=1 under --ref=cpu"
        )
    C = shape[1]
    if affine:
        weight = torch.randn(C, dtype=dtype, device=flag_gems.device)
        bias = torch.randn(C, dtype=dtype, device=flag_gems.device)
    else:
        weight = None
        bias = None

    (
        inp,
        save_mean,
        save_var,
        running_mean,
        running_var,
        reserve,
    ) = _forward_save_stats(
        shape, dtype, flag_gems.device, weight, bias, True, impl_index
    )

    grad_output = torch.randn_like(inp)

    if affine:
        output_mask = [True, True, True]
    else:
        output_mask = [True, False, False]

    # The cudnn backward path (impl_index == 1) requires a real weight tensor,
    # so emulate "no affine" with a neutral ones weight in that case.  The
    # native path (impl_index == 0) accepts ``None`` and treats it as weight 1.
    if weight is None and impl_index == 1:
        backward_weight = torch.ones(C, dtype=dtype, device=flag_gems.device)
    else:
        backward_weight = weight

    ref_inp = utils.to_reference(inp, True)
    ref_grad = utils.to_reference(grad_output, True)
    ref_weight = utils.to_reference(backward_weight, True)
    ref_running_mean = utils.to_reference(running_mean, True)
    ref_running_var = utils.to_reference(running_var, True)
    ref_save_mean = utils.to_reference(save_mean, True)
    ref_save_var = utils.to_reference(save_var, True)
    ref_reserve = utils.to_reference(reserve, True)

    fn = torch.ops.aten._batch_norm_impl_index_backward
    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = fn(
        impl_index,
        ref_inp,
        ref_grad,
        ref_weight,
        ref_running_mean,
        ref_running_var,
        ref_save_mean,
        ref_save_var,
        True,
        1e-5,
        output_mask,
        ref_reserve,
    )

    with flag_gems.use_gems():
        (
            res_in_grad,
            res_weight_grad,
            res_bias_grad,
        ) = fn(
            impl_index,
            inp,
            grad_output,
            backward_weight,
            running_mean,
            running_var,
            save_mean,
            save_var,
            True,
            1e-5,
            output_mask,
            reserve,
        )

    reduce_dim = math.prod(shape) // C
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=reduce_dim)
    if affine:
        utils.gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=reduce_dim
        )
        utils.gems_assert_close(
            res_bias_grad, ref_bias_grad, dtype, reduce_dim=reduce_dim
        )


@pytest.mark.batch_norm_impl_index_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("impl_index", [0, 1])
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_impl_index_backward_eval(shape, dtype, impl_index, affine):
    """Eval-mode backward: saved stats are empty and the inverse std is rebuilt
    from ``running_var``; the input gradient degenerates to
    ``grad_output * weight * inv_std``."""
    C = shape[1]
    if affine:
        weight = torch.randn(C, dtype=dtype, device=flag_gems.device)
    else:
        weight = None

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    running_mean = torch.zeros(C, dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(C, dtype=dtype, device=flag_gems.device)
    # In eval mode the forward returns empty saved statistics.
    save_mean = torch.empty(0, dtype=dtype, device=flag_gems.device)
    save_var = torch.empty(0, dtype=dtype, device=flag_gems.device)
    reserve = torch.empty(0, dtype=torch.uint8, device=flag_gems.device)

    grad_output = torch.randn_like(inp)

    if affine:
        output_mask = [True, True, True]
    else:
        output_mask = [True, False, False]

    ref_inp = utils.to_reference(inp, True)
    ref_grad = utils.to_reference(grad_output, True)
    ref_weight = utils.to_reference(weight, True)
    ref_running_mean = utils.to_reference(running_mean, True)
    ref_running_var = utils.to_reference(running_var, True)
    ref_save_mean = utils.to_reference(save_mean, True)
    ref_save_var = utils.to_reference(save_var, True)
    ref_reserve = utils.to_reference(reserve, True)

    fn = torch.ops.aten._batch_norm_impl_index_backward
    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = fn(
        impl_index,
        ref_inp,
        ref_grad,
        ref_weight,
        ref_running_mean,
        ref_running_var,
        ref_save_mean,
        ref_save_var,
        False,
        1e-5,
        output_mask,
        ref_reserve,
    )

    with flag_gems.use_gems():
        (
            res_in_grad,
            res_weight_grad,
            res_bias_grad,
        ) = fn(
            impl_index,
            inp,
            grad_output,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_var,
            False,
            1e-5,
            output_mask,
            reserve,
        )

    reduce_dim = math.prod(shape) // C
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=reduce_dim)
    if affine:
        utils.gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=reduce_dim
        )
        utils.gems_assert_close(
            res_bias_grad, ref_bias_grad, dtype, reduce_dim=reduce_dim
        )


@pytest.mark.batch_norm_impl_index_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_impl_index_backward_empty(dtype):
    """Empty-input edge case: zero-element gradients must keep the autograd
    graph intact."""
    # Empty NCHW input (batch dim 0) to exercise the zero-element backward path.
    shape = (0, 3, 4, 5)
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn_like(inp)
    weight = torch.randn(C, dtype=dtype, device=flag_gems.device)
    running_mean = torch.zeros(C, dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(C, dtype=dtype, device=flag_gems.device)
    save_mean = torch.empty(0, dtype=dtype, device=flag_gems.device)
    save_var = torch.empty(0, dtype=dtype, device=flag_gems.device)
    reserve = torch.empty(0, dtype=torch.uint8, device=flag_gems.device)

    output_mask = [True, True, True]

    ref_inp = utils.to_reference(inp, True)
    ref_grad = utils.to_reference(grad_output, True)
    ref_weight = utils.to_reference(weight, True)
    ref_running_mean = utils.to_reference(running_mean, True)
    ref_running_var = utils.to_reference(running_var, True)
    ref_save_mean = utils.to_reference(save_mean, True)
    ref_save_var = utils.to_reference(save_var, True)
    ref_reserve = utils.to_reference(reserve, True)

    fn = torch.ops.aten._batch_norm_impl_index_backward
    ref_out = fn(
        0,
        ref_inp,
        ref_grad,
        ref_weight,
        ref_running_mean,
        ref_running_var,
        ref_save_mean,
        ref_save_var,
        True,
        1e-5,
        output_mask,
        ref_reserve,
    )

    with flag_gems.use_gems():
        res_out = fn(
            0,
            inp,
            grad_output,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_var,
            True,
            1e-5,
            output_mask,
            reserve,
        )

    for res, ref in zip(res_out, ref_out):
        assert res.shape == ref.shape
    # weight/bias gradients over an empty reduction are zero.
    utils.gems_assert_close(res_out[1], ref_out[1], dtype, reduce_dim=1)
    utils.gems_assert_close(res_out[2], ref_out[2], dtype, reduce_dim=1)
