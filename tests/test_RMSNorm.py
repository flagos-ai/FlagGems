import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.RMSNorm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_RMSNorm(shape, dtype):
    """Test RMSNorm with simple interface (no explicit weight)."""
    # RMSNorm normalizes over the last dimension
    N = shape[1]

    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape[:2]).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device, requires_grad=True)
    eps = 1e-5

    ref_inp = to_reference(inp)

    # Reference implementation using torch.nn.RMSNorm
    rms_norm_module = torch.nn.RMSNorm(N, eps=eps, elementwise_affine=False)
    ref_out = rms_norm_module(ref_inp)

    # Test with flag_gems - signature: RMSNorm(input, normalized_shape, weight=None, eps=1e-5)
    with flag_gems.use_gems():
        res_out = flag_gems.RMSNorm(inp, (N,), None, eps)

    # Test gradients
    res_grad = torch.tensor(
        np_grad, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_grad = to_reference(res_grad)

    res_gradients = torch.autograd.grad(res_out, inp, res_grad)
    ref_gradients = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    gems_assert_close(res_out, ref_out, dtype)
    gems_assert_close(res_gradients[0], ref_gradients[0], dtype)


@pytest.mark.RMSNorm_
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_RMSNorm_inplace(shape, dtype):
    """Test in-place RMSNorm."""
    N = shape[1]

    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone())

    eps = 1e-5

    # Reference: use nn.RMSNorm module then copy back
    rms_norm_module = torch.nn.RMSNorm(N, eps=eps, elementwise_affine=False)
    ref_out = rms_norm_module(ref_inp)

    # Test in-place with flag_gems - signature: RMSNorm_(input, normalized_shape, eps=1e-5)
    with flag_gems.use_gems():
        res_out = flag_gems.RMSNorm_(inp, (N,), eps)

    gems_assert_close(res_out, ref_out, dtype)
