# AMIN operator test

import os
import sys

import pytest
import torch

import flag_gems
from flag_gems.experimental_ops.amin import amin as gems_amin
from flag_gems.experimental_ops.amin import amin_out as gems_amin_out

# Add parent directory to path to import flag_gems
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
try:
    from tests.accuracy_utils import TO_CPU, gems_assert_close  # noqa: E402
except ImportError:
    # Fallback values when running outside pytest
    TO_CPU = False  # fallback

    def gems_assert_close(res, ref, dtype, **kwargs):
        # Simple fallback comparison
        torch.testing.assert_close(res, ref, **kwargs)


def to_reference(inp, upcast=False):
    if inp is None:
        return None
    if TO_CPU:
        ref_inp = inp.to("cpu")
    else:
        ref_inp = inp.clone()
    if upcast:
        if ref_inp.is_complex():
            ref_inp = ref_inp.to(torch.complex128)
        else:
            ref_inp = ref_inp.to(torch.float64)
    return ref_inp


@pytest.mark.amin
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 320)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [None, 0, 1, -1, [0, 1]])
@pytest.mark.parametrize("keepdim", [False, True])
def test_amin_tensor_reduce_2d(shape, dtype, dim, keepdim):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    if dim is None and not keepdim:
        ref_out = torch.ops.aten.amin(ref_x)
        with flag_gems.use_gems():
            act_out = gems_amin(x)
    else:
        use_dim = list(range(len(shape))) if dim is None else dim
        ref_out = torch.ops.aten.amin(ref_x, use_dim, keepdim)
        with flag_gems.use_gems():
            act_out = gems_amin(x, use_dim, keepdim)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.amin
@pytest.mark.parametrize("shape", [(2, 3, 4), (16, 17, 8), (32, 64, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [None, 0, 1, 2, -1, [0, 2], [1, 2], [0, 1, 2]])
@pytest.mark.parametrize("keepdim", [False, True])
def test_amin_tensor_reduce_3d(shape, dtype, dim, keepdim):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    if dim is None and not keepdim:
        ref_out = torch.ops.aten.amin(ref_x)
        with flag_gems.use_gems():
            act_out = gems_amin(x)
    else:
        use_dim = list(range(len(shape))) if dim is None else dim
        ref_out = torch.ops.aten.amin(ref_x, use_dim, keepdim)
        with flag_gems.use_gems():
            act_out = gems_amin(x, use_dim, keepdim)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.amin
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 320)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [None, 0, 1, -1, [0, 1]])
@pytest.mark.parametrize("keepdim", [False, True])
def test_amin_out_reduce_2d(shape, dtype, dim, keepdim):
    def test_compute_out_shape(shape, dims, keepdim):
        if dims is None:
            # reduce-all default with keepdim=False
            return ()
        if isinstance(dims, int):
            dims = [dims]
        dims = [(d + len(shape)) % len(shape) for d in dims]
        if keepdim:
            out_shape = list(shape)
            for d in dims:
                out_shape[d] = 1
            return tuple(out_shape)
        else:
            remaining = [i for i in range(len(shape)) if i not in set(dims)]
            return tuple(shape[i] for i in remaining)

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    if dim is None and not keepdim:
        out_shape = test_compute_out_shape(shape, None, keepdim)
        ref_out_t = torch.empty(out_shape, dtype=dtype, device=ref_x.device)
        act_out_t = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

        ref_out = torch.ops.aten.amin.out(ref_x, out=ref_out_t)
        with flag_gems.use_gems():
            act_out = gems_amin_out(x, act_out_t)
    else:
        use_dim = list(range(len(shape))) if dim is None else dim
        out_shape = test_compute_out_shape(shape, use_dim, keepdim)
        ref_out_t = torch.empty(out_shape, dtype=dtype, device=ref_x.device)
        act_out_t = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

        ref_out = torch.ops.aten.amin.out(ref_x, use_dim, keepdim, out=ref_out_t)
        with flag_gems.use_gems():
            act_out = gems_amin_out(x, use_dim, keepdim, act_out_t)

    gems_assert_close(act_out, ref_out, dtype=dtype)


@pytest.mark.amin
@pytest.mark.parametrize("shape", [(2, 3, 4), (16, 17, 8), (32, 64, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [None, 0, 1, 2, -1, [0, 2], [1, 2], [0, 1, 2]])
@pytest.mark.parametrize("keepdim", [False, True])
def test_amin_out_reduce_3d(shape, dtype, dim, keepdim):
    def test_compute_out_shape(shape, dims, keepdim):
        if dims is None:
            return ()
        if isinstance(dims, int):
            dims = [dims]
        dims = [(d + len(shape)) % len(shape) for d in dims]
        if keepdim:
            out_shape = list(shape)
            for d in dims:
                out_shape[d] = 1
            return tuple(out_shape)
        else:
            remaining = [i for i in range(len(shape)) if i not in set(dims)]
            return tuple(shape[i] for i in remaining)

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = to_reference(x)

    if dim is None and not keepdim:
        out_shape = test_compute_out_shape(shape, None, keepdim)
        ref_out_t = torch.empty(out_shape, dtype=dtype, device=ref_x.device)
        act_out_t = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

        ref_out = torch.ops.aten.amin.out(ref_x, out=ref_out_t)
        with flag_gems.use_gems():
            act_out = gems_amin_out(x, act_out_t)
    else:
        use_dim = list(range(len(shape))) if dim is None else dim
        out_shape = test_compute_out_shape(shape, use_dim, keepdim)
        ref_out_t = torch.empty(out_shape, dtype=dtype, device=ref_x.device)
        act_out_t = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

        ref_out = torch.ops.aten.amin.out(ref_x, use_dim, keepdim, out=ref_out_t)
        with flag_gems.use_gems():
            act_out = gems_amin_out(x, use_dim, keepdim, act_out_t)

    gems_assert_close(act_out, ref_out, dtype=dtype)
