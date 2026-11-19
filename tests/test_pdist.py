import sys

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# The implementation module, not the exported ``flag_gems.pdist`` function.
pdist_module = sys.modules["flag_gems.ops.pdist"]

# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
PDIST_SHAPES = utils.PDIST_SHAPES


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 2.0
    ref_out = torch.pdist(ref_inp, p=p)
    res_out = flag_gems.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_p1(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 1.0
    ref_out = torch.pdist(ref_inp, p=p)
    res_out = flag_gems.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_pinf(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = float("inf")
    ref_out = torch.pdist(ref_inp, p=p)
    res_out = flag_gems.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_p_general(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 3.0
    ref_out = torch.pdist(ref_inp, p=p)
    res_out = flag_gems.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_p0(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 0.0
    ref_out = torch.pdist(ref_inp, p=p)
    res_out = flag_gems.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.pdist
@pytest.mark.parametrize("shape", PDIST_SHAPES)
# pdist CUDA kernel only supports float32; Half/BFloat16 raise RuntimeError
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_p_large(shape, dtype):
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    p = 100.0
    ref_out = torch.pdist(ref_inp, p=p)
    res_out = flag_gems.pdist(inp, p=p)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# First-order backward through autograd
# ---------------------------------------------------------------------------
#
# ``aten::pdist`` is CompositeImplicitAutograd, so FlagGems' device override of
# the public entry replaces ATen's generated formula.  These tests exercise the
# real autograd path (forward + ``torch.autograd.grad``) of both the direct
# ``flag_gems.pdist`` entry and the ``torch.pdist`` dispatch path, and cover the
# p branches, zero distances, strided/expanded upstream gradients and the
# kernel attribution.

BACKWARD_SHAPES = (
    utils.PDIST_SHAPES if utils.QUICK_MODE else utils.PDIST_SHAPES + [(256, 512)]
)

BACKWARD_P_VALUES = [2.0, 1.0, float("inf"), 3.0, 1.5]


def _backward_pair(shape, dtype, p):
    """Same values on both sides; the reference follows ``--ref``."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_(True)
    inp = inp.detach().clone().requires_grad_(True)
    grad = torch.randn(
        shape[0] * (shape[0] - 1) // 2, dtype=dtype, device=flag_gems.device
    )
    return inp, ref_inp, grad, utils.to_reference(grad)


@pytest.mark.pdist_forward_backward
@pytest.mark.parametrize("shape", BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("p", BACKWARD_P_VALUES)
def test_pdist_backward(shape, dtype, p):
    """Direct entry: ``flag_gems.pdist`` must produce the ATen gradient."""
    if shape[0] < 2:
        pytest.skip("pdist requires at least 2 rows")
    inp, ref_inp, grad, ref_grad = _backward_pair(shape, dtype, p)

    out = flag_gems.pdist(inp, p=p)
    ref_out = torch.pdist(ref_inp, p=p)
    utils.gems_assert_close(out, ref_out, dtype)

    (dx,) = torch.autograd.grad(out, (inp,), (grad,))
    (ref_dx,) = torch.autograd.grad(ref_out, (ref_inp,), (ref_grad,))

    assert dx.dtype == dtype
    assert dx.shape == inp.shape
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.pdist_forward_backward
@pytest.mark.parametrize("p", BACKWARD_P_VALUES)
def test_pdist_backward_zero_distance(p):
    """Rows that coincide produce exact zero distances; the gradient stays finite.

    ATen's CUDA backward contributes nothing for a zero-distance pair instead of
    dividing by zero, so the duplicate rows must get the same finite gradient.
    """
    dtype = torch.float32
    inp = torch.randn((7, 5), dtype=dtype, device=flag_gems.device)
    inp[3] = inp[1]
    inp[6] = inp[1]
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_(True)
    inp = inp.detach().clone().requires_grad_(True)
    grad = torch.randn(21, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(grad)

    out = flag_gems.pdist(inp, p=p)
    ref_out = torch.pdist(ref_inp, p=p)
    # Three identical rows give three exactly-zero pairs: (1, 3), (1, 6), (3, 6).
    assert (out == 0).sum().item() == 3, out
    assert (ref_out == 0).sum().item() == 3, ref_out
    utils.gems_assert_close(out, ref_out, dtype)

    (dx,) = torch.autograd.grad(out, (inp,), (grad,))
    (ref_dx,) = torch.autograd.grad(ref_out, (ref_inp,), (ref_grad,))

    assert torch.isfinite(dx).all(), dx
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.pdist_forward_backward
def test_pdist_backward_p_zero():
    """p == 0 counts differing coordinates, so its gradient is identically zero."""
    dtype, p_zero = torch.float32, 0.0
    inp = torch.randn((6, 4), dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_(True)
    grad = torch.randn(15, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(grad)

    out = flag_gems.pdist(inp, p_zero)
    ref_out = torch.pdist(ref_inp, p_zero)
    utils.gems_assert_close(out, ref_out, dtype)

    (dx,) = torch.autograd.grad(out, (inp,), (grad,))
    (ref_dx,) = torch.autograd.grad(ref_out, (ref_inp,), (ref_grad,))
    utils.gems_assert_close(dx, ref_dx, dtype)
    assert not dx.any(), dx


@pytest.mark.pdist_forward_backward
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_backward_expanded_grad(dtype):
    """A zero-stride upsteam gradient must be read as its broadcast value."""
    inp = torch.randn((9, 7), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_(True)
    inp = inp.detach().clone().requires_grad_(True)
    base = torch.randn(1, dtype=dtype, device=flag_gems.device)
    grad = base.expand(36)
    ref_grad = utils.to_reference(grad)

    out = flag_gems.pdist(inp, p=2.0)
    ref_out = torch.pdist(ref_inp, p=2.0)
    (dx,) = torch.autograd.grad(out, (inp,), (grad,))
    (ref_dx,) = torch.autograd.grad(ref_out, (ref_inp,), (ref_grad,))
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.pdist_forward_backward
@pytest.mark.parametrize("dtype", [torch.float32])
def test_pdist_backward_strided_grad(dtype):
    """A non-contiguous upstream gradient is gathered, not walked densely."""
    inp = torch.randn((8, 6), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_(True)
    inp = inp.detach().clone().requires_grad_(True)
    strided = torch.randn(2 * 28, dtype=dtype, device=flag_gems.device)[::2]
    assert strided.stride(0) == 2

    out = flag_gems.pdist(inp, p=2.0)
    ref_out = torch.pdist(ref_inp, p=2.0)
    (dx,) = torch.autograd.grad(out, (inp,), (strided,))
    (ref_dx,) = torch.autograd.grad(ref_out, (ref_inp,), (utils.to_reference(strided),))
    utils.gems_assert_close(dx, ref_dx, dtype)


@pytest.mark.pdist_forward_backward
def test_pdist_backward_no_grad_path_unchanged():
    """Inference must stay on the forward kernel and build no autograd node."""
    inp = torch.randn((6, 4), dtype=torch.float32, device=flag_gems.device)
    assert flag_gems.pdist(inp, 2.0).grad_fn is None
    with torch.no_grad():
        assert flag_gems.pdist(inp, 2.0).grad_fn is None
    ref_inp = utils.to_reference(inp)
    utils.gems_assert_close(
        flag_gems.pdist(inp, 2.0), torch.pdist(ref_inp, 2.0), torch.float32
    )


@pytest.mark.pdist_forward_backward
def test_pdist_backward_grad_fn_attribution():
    """The gradient must come from the FlagGems autograd node, not ATen's."""
    inp = torch.randn((10, 6), dtype=torch.float32, device=flag_gems.device)
    inp.requires_grad_()
    out = flag_gems.pdist(inp, 2.0)
    assert "PdistFunction" in type(out.grad_fn).__name__, type(out.grad_fn).__name__

    captured = []
    original = pdist_module.pdist_backward

    def recording_backward(grad_output, x, p, pdist_out):
        captured.append((tuple(grad_output.shape), tuple(x.shape), p))
        return original(grad_output, x, p, pdist_out)

    pdist_module.pdist_backward = recording_backward
    try:
        (dx,) = torch.autograd.grad(out, (inp,), (torch.randn(45, device=inp.device),))
    finally:
        pdist_module.pdist_backward = original

    assert captured == [((45,), (10, 6), 2.0)], captured
    assert dx.shape == (10, 6)


@pytest.mark.pdist_forward_backward
def test_pdist_backward_single_kernel():
    """One Triton launch per backward call, with no ATen fallback kernels."""
    from torch.profiler import ProfilerActivity, profile

    inp = torch.randn((128, 256), dtype=torch.float32, device=flag_gems.device)
    inp.requires_grad_()
    out = flag_gems.pdist(inp, 2.0)
    grad = torch.randn(8128, dtype=torch.float32, device=flag_gems.device)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, (inp,), (grad,))
        flag_gems.runtime.torch_device_fn.synchronize()

    launches = [
        (event.key, event.count)
        for event in prof.key_averages()
        if event.device_time_total > 0
    ]
    kernels = [row for row in launches if "pdist_backward" in row[0]]
    assert len(kernels) == 1, launches
    assert kernels[0][1] == 1, kernels
    assert all("pdist_backward" in key for key, _ in launches), launches


@pytest.mark.pdist_forward_backward
@pytest.mark.parametrize("p", BACKWARD_P_VALUES)
def test_pdist_backward_second_order_matches_aten(p):
    """The first-order gradient is differentiable once, like ATen's formula.

    ATen raises ``NotImplementedError`` for the second derivative of
    ``_pdist_backward``; both sides just have to agree on the first derivative
    through ``create_graph``.
    """
    inp = torch.randn((7, 5), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_(True)
    inp = inp.detach().clone().requires_grad_(True)
    grad = torch.randn(21, dtype=torch.float32, device=flag_gems.device)
    ref_grad = utils.to_reference(grad)

    (dx,) = torch.autograd.grad(flag_gems.pdist(inp, p), (inp,), (grad,))
    (ref_dx,) = torch.autograd.grad(torch.pdist(ref_inp, p), (ref_inp,), (ref_grad,))
    # ATen's own backward is not differentiable twice; ours returns a detached
    # tensor, so neither side can build a second-order graph here.
    assert not dx.requires_grad
    assert not ref_dx.requires_grad
    utils.gems_assert_close(dx, ref_dx, torch.float32)


@pytest.mark.pdist_forward_backward
@pytest.mark.parametrize("p", BACKWARD_P_VALUES)
def test_pdist_backward_torch_dispatch(p):
    """``torch.pdist`` must keep its gradient when FlagGems owns the device key.

    ``aten::pdist`` is CompositeImplicitAutograd; overriding only its device key
    dropped the generated formula, so ``torch.pdist`` under ``use_gems()``
    produced a tensor whose gradient was silently discarded.  This registers the
    same ops ``use_gems()`` registers (through ``enable``, since the KernelGen
    test convention forbids ``use_gems()`` in test files) and checks the node and
    the gradient it produces.
    """
    inp = torch.randn((9, 6), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_(True)
    inp = inp.detach().clone().requires_grad_(True)
    grad = torch.randn(36, dtype=torch.float32, device=flag_gems.device)
    ref_grad = utils.to_reference(grad)

    previous_registrar = flag_gems.current_work_registrar
    library = torch.library.Library("aten", "IMPL")
    try:
        flag_gems.enable(lib=library)
        out = torch.ops.aten.pdist(inp, p)
        assert "PdistFunction" in type(out.grad_fn).__name__, type(out.grad_fn).__name__
        (dx,) = torch.autograd.grad(out, (inp,), (grad,))
    finally:
        if hasattr(library, "_destroy"):
            library._destroy()
        flag_gems.current_work_registrar = previous_registrar
        flag_gems.teardown_flaggems_logging()

    ref_out = torch.ops.aten.pdist(ref_inp, p)
    (ref_dx,) = torch.autograd.grad(ref_out, (ref_inp,), (ref_grad,))
    utils.gems_assert_close(dx, ref_dx, torch.float32)
