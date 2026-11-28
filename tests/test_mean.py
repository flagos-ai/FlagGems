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
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
    KEEPDIM_DIMS = [(True, 1)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    KEEPDIM_DIMS = list(zip([True, False] * 2, DIMS_LIST))


@pytest.mark.mean
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp)
    res_out = flag_gems.mean(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.mean
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_empty(shape, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.mean(inp)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.mean
def test_mean_empty_explicit_dtype():
    inp = torch.empty((0,), dtype=torch.float16, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dtype=torch.float32)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dtype=torch.float32)

    assert res_out.dtype == torch.float32
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.mean
@pytest.mark.parametrize("dtype", [torch.int32, torch.bool])
def test_mean_empty_rejects_non_floating_input(dtype):
    inp = torch.empty((0,), dtype=dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.mean(inp)


@pytest.mark.mean
def test_mean_empty_rejects_non_floating_dtype():
    inp = torch.empty((0,), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.mean(inp, dtype=torch.int32)


@pytest.mark.mean_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mean_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    res_out = flag_gems.mean_dim(inp, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)


MEAN_EMPTY_DIM_CASES = [
    ((2, 0), 1),
    ((0, 3), 0),
    ((0, 3), 1),
    ((2, 0), -1),
    ((2, 0, 3), [0, 1]),
    ((2, 0, 3), [1, 2]),
    ((0, 2, 3), [1, 2]),
]


@pytest.mark.mean_dim
@pytest.mark.parametrize("shape, dim", MEAN_EMPTY_DIM_CASES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mean_dim_empty(shape, dim, keepdim, dtype):
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim, keepdim)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.mean_dim
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mean_dim_empty_explicit_dtype():
    inp = torch.empty((2, 0), dtype=torch.float16, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim=1, dtype=torch.float32)
    with flag_gems.use_gems():
        res_out = torch.mean(inp, dim=1, dtype=torch.float32)

    assert res_out.dtype == torch.float32
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


# Shapes where K (product of dims after the reduction axis) exceeds the CUDA
# grid-Y limit of 65535, which used to trigger "Triton Error [CUDA]: invalid
# argument" before the mean_heur_tile_k grid-overflow fix.
MEAN_LARGE_K_SHAPES = [
    (1, 8, 256, 256),  # dim=1 → M=1, N=8, K=65536 (just over limit)
    (1, 4, 512, 512),  # dim=1 → M=1, N=4, K=262144 (well over limit)
]


@pytest.mark.mean_dim
@pytest.mark.parametrize("shape", MEAN_LARGE_K_SHAPES)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mean_dim_large_k(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    res_out = flag_gems.mean_dim(inp, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)


MEAN_LARGE_INNERDIM_SHAPES = [
    (1024, 1024, 1024),  # dim=1 → M=1, N=8, K=65536 (just over limit)
    (1024, 2048, 1024),  # dim=1 → M=1, N=4, K=262144 (well over limit)
]


@pytest.mark.mean_dim
@pytest.mark.parametrize("shape", MEAN_LARGE_INNERDIM_SHAPES)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_mean_dim_large_innerdim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, dim, keepdim)
    res_out = flag_gems.mean_dim(inp, dim, keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype)


MEAN_BACKWARD_SHAPES = utils.REDUCTION_SHAPES + [
    (),
    (0,),
    (2, 0, 3),
]


def _mean_forward(api, inp, dim, keepdim, dtype_arg=None):
    if dim is None:
        out = api(inp, dtype=dtype_arg) if dtype_arg is not None else api(inp)
        return out.reshape([1] * inp.dim()) if keepdim else out
    if dtype_arg is not None:
        return api(inp, dim, keepdim, dtype=dtype_arg)
    return api(inp, dim, keepdim)


def _mean_backward_check(
    shape, dtype, dim=None, keepdim=False, dtype_arg=None, inp=None
):
    if inp is None:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp.requires_grad_(True)
    ref_inp = utils.to_reference(inp, True)

    res_out = _mean_forward(
        flag_gems.mean_dim if dim is not None else flag_gems.mean,
        inp,
        dim,
        keepdim,
        dtype_arg,
    )
    ref_out = _mean_forward(torch.mean, ref_inp, dim, keepdim, dtype_arg)
    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, res_out.dtype, equal_nan=True)
    assert res_out.grad_fn is not None

    out_grad = torch.randn(res_out.shape, dtype=res_out.dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad, True)
    ref_grad_inp = torch.autograd.grad(ref_out, ref_inp, ref_grad)[0]
    res_grad_inp = torch.autograd.grad(res_out, inp, out_grad)[0]

    assert res_grad_inp.shape == ref_grad_inp.shape
    assert res_grad_inp.dtype == dtype
    utils.gems_assert_close(res_grad_inp, ref_grad_inp, dtype, equal_nan=True)
    return res_out, res_grad_inp


@pytest.mark.mean_backward
@pytest.mark.parametrize("shape", MEAN_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_backward(shape, dtype):
    res_out, _ = _mean_backward_check(shape, dtype)
    assert "MeanFunction" in res_out.grad_fn.name()


@pytest.mark.mean_backward
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_backward_keepdim_all_reduce(keepdim, dtype):
    # Small exact values isolate shape and gradient handling from the inherited
    # low-precision forward's rounding of partial sums after cancellation.
    inp = (
        torch.arange(64, device=flag_gems.device).remainder(4).reshape(2, 32).to(dtype)
    )
    _mean_backward_check(inp.shape, dtype, keepdim=keepdim, inp=inp)


@pytest.mark.mean_backward
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES + [(1, 8, 256, 256)])
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIMS + [(True, 1), (False, 1)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_dim_backward(shape, dim, keepdim, dtype):
    _mean_backward_check(shape, dtype, dim=dim, keepdim=keepdim)


@pytest.mark.mean_backward
@pytest.mark.parametrize("shape, dim", MEAN_EMPTY_DIM_CASES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_dim_backward_empty(shape, dim, keepdim, dtype):
    _mean_backward_check(shape, dtype, dim=dim, keepdim=keepdim)


@pytest.mark.mean_backward
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mean_backward_explicit_dtype(shape, dtype):
    res_out, _ = _mean_backward_check(shape, dtype, dtype_arg=torch.float32)
    assert res_out.dtype == torch.float32


@pytest.mark.mean_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_backward_non_contiguous(dtype):
    base = torch.randn((8, 64, 32), dtype=dtype, device=flag_gems.device)
    inp = base.permute(1, 0, 2)
    assert not inp.is_contiguous()
    inp.requires_grad_(True)
    ref_inp = utils.to_reference(inp, True)

    res_out = flag_gems.mean_dim(inp, 1, True)
    ref_out = torch.mean(ref_inp, 1, True)
    out_grad = torch.randn((32, 64, 1), dtype=dtype, device=flag_gems.device)
    out_grad = out_grad.permute(1, 2, 0)
    assert not out_grad.is_contiguous()
    ref_grad = utils.to_reference(out_grad, True)

    ref_grad_inp = torch.autograd.grad(ref_out, ref_inp, ref_grad)[0]
    res_grad_inp = torch.autograd.grad(res_out, inp, out_grad)[0]
    assert res_grad_inp.shape == ref_grad_inp.shape
    utils.gems_assert_close(res_grad_inp, ref_grad_inp, dtype, equal_nan=True)


@pytest.mark.mean_backward
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", [0, 1, [0, 1], [1, 0], None])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_backward_use_gems(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp.requires_grad_(True)
    ref_inp = utils.to_reference(inp, True)

    res_out = (
        flag_gems.mean(inp) if dim is None else flag_gems.mean_dim(inp, dim, False)
    )
    ref_out = torch.mean(ref_inp) if dim is None else torch.mean(ref_inp, dim, False)
    utils.gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn(res_out.shape, dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(out_grad, True)
    ref_grad_inp = torch.autograd.grad(ref_out, ref_inp, ref_grad)[0]
    res_grad_inp = torch.autograd.grad(res_out, inp, out_grad)[0]

    assert res_grad_inp.shape == ref_grad_inp.shape
    assert res_grad_inp.dtype == dtype
    utils.gems_assert_close(res_grad_inp, ref_grad_inp, dtype, equal_nan=True)


@pytest.mark.mean_backward
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_mean_backward_no_grad_input(dtype):
    inp = torch.randn((16, 16), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.mean(ref_inp, 1)
    res_out = flag_gems.mean_dim(inp, 1)
    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out.grad_fn is None


@pytest.mark.mean_backward
@pytest.mark.parametrize("dim", [None, 0, 1, [0, 1]])
def test_mean_backward_cpu_keeps_native_path(dim):
    # The backward is wired to this backend's device autograd key only, so CPU
    # tensors must keep torch's native mean autograd even inside use_gems().
    inp = torch.randn(8, 8, requires_grad=True)
    res_out = torch.mean(inp) if dim is None else torch.mean(inp, dim)
    assert type(res_out.grad_fn).__name__ in {"MeanBackward0", "MeanBackward1"}
    out_grad = torch.randn(res_out.shape)
    assert torch.autograd.grad(res_out, inp, out_grad)[0].shape == inp.shape


@pytest.mark.mean_backward
@pytest.mark.parametrize("dim", [None, 0])
@pytest.mark.parametrize(
    "dtype, out_dtype", [(torch.complex64, None), (torch.float32, torch.complex64)]
)
def test_mean_backward_rejects_complex(dim, dtype, out_dtype):
    inp = torch.ones((2, 3), dtype=dtype, device=flag_gems.device, requires_grad=True)
    with pytest.raises(NotImplementedError, match="complex"):
        if dim is None:
            flag_gems.mean(inp, dtype=out_dtype)
        else:
            flag_gems.mean_dim(inp, dim, dtype=out_dtype)


@pytest.mark.mean_backward
@pytest.mark.parametrize(
    "grad_dtype, input_dtype",
    [(torch.complex64, torch.float32), (torch.float32, torch.complex64)],
)
def test_mean_backward_public_rejects_complex(grad_dtype, input_dtype):
    grad_output = torch.ones((), dtype=grad_dtype, device="cpu")
    with pytest.raises(NotImplementedError, match="complex"):
        flag_gems.mean_backward(grad_output, (2, 3), input_dtype)
