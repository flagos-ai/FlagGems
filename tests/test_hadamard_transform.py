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

import math

import pytest
import torch
import torch.nn.functional as F
import triton
from packaging.version import Version

import flag_gems
from tests.hadamard_reference import special_hadamard_reference

from . import accuracy_utils as utils
from . import conftest as cfg

try:
    from scipy.linalg import hadamard as scipy_hadamard
except ImportError:  # pragma: no cover
    scipy_hadamard = None

_TRITON_VERSION = Version(triton.__version__.split("+")[0])
_SKIP_JOIN_BUG = _TRITON_VERSION < Version("3.5.0")
_skip_if_join_bug = pytest.mark.skipif(
    _SKIP_JOIN_BUG,
    reason=f"triton {triton.__version__} has tt.join layout bug (fixed in 3.5.0)",
)

# Standard FHT dimensions (exclude 16384/32768: scipy full matrix is too large)
if cfg.QUICK_MODE:
    HADAMARD_DIMS = [64, 137, 256, 512, 1024]
else:
    HADAMARD_DIMS = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        137,
        1024,
        2048,
        4096,
        8192,
    ]

if cfg.QUICK_MODE:
    HADAMARD_MN_CASES = [
        (1536, 12, "12N"),
        (10240, 20, "20N"),
        (14336, 28, "28N"),
        (40960, 40, "40N"),
    ]
else:
    HADAMARD_MN_CASES = [
        (1536, 12, "12N"),
        (3072, 12, "12N"),
        (6144, 12, "12N"),
        (12288, 12, "12N"),
        (10240, 20, "20N"),
        (20480, 20, "20N"),
        (14336, 28, "28N"),
        (20480, 40, "40N"),
        (40960, 40, "40N"),
    ]

_FN_MAP = {
    "12N": flag_gems.fast_hadamard_transform_12N,
    "20N": flag_gems.fast_hadamard_transform_20N,
    "28N": flag_gems.fast_hadamard_transform_28N,
    "40N": flag_gems.fast_hadamard_transform_40N,
}


def _hadamard_transform_ref(x, scale=1.0):
    """Reference using a SciPy Hadamard matrix with padding and cropping."""
    if scipy_hadamard is None:
        raise ImportError("Please install scipy")
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim)) if dim > 0 else 0
    dim_padded = 1 << log_dim if dim > 0 else 1
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(
        x,
        torch.tensor(
            scipy_hadamard(dim_padded, dtype=float),
            dtype=x.dtype,
            device=x.device,
        ),
    )
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


@pytest.mark.fast_hadamard_transform
@_skip_if_join_bug
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", HADAMARD_DIMS)
def test_fast_hadamard_transform(dim, dtype):
    """Compare against a SciPy Hadamard matrix multiply in fp32."""
    if scipy_hadamard is None:
        pytest.skip("scipy is required for hadamard_transform_ref")

    atol = 3e-3 if dtype == torch.float32 else 5e-3
    if dtype == torch.bfloat16:
        atol = 5e-2

    torch.random.manual_seed(0)
    batch_size = 15
    device = flag_gems.device
    x = torch.randn(batch_size, dim, device=device, dtype=dtype).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    x_pt = x.detach().clone().requires_grad_()
    scale = 1 / math.sqrt(dim)

    out = flag_gems.fast_hadamard_transform(x, scale=scale)
    out_ref = _hadamard_transform_ref(x_ref.float(), scale=scale)
    out_pt = _hadamard_transform_ref(x_pt, scale=scale)

    out_err = (out.float() - out_ref).abs().max().item()
    pt_err = (out_pt.float() - out_ref).abs().max().item()
    assert (
        out_err < 2 * pt_err + atol
    ), f"forward dim={dim} dtype={dtype}: out_err={out_err}, pt_err={pt_err}, atol={atol}"

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    out_pt.backward(g)

    dx_err = (x.grad.float() - x_ref.grad.float()).abs().max().item()
    dx_pt_err = (x_pt.grad.float() - x_ref.grad.float()).abs().max().item()
    assert (
        dx_err < 2 * dx_pt_err + atol
    ), f"backward dim={dim} dtype={dtype}: dx_err={dx_err}, dx_pt_err={dx_pt_err}, atol={atol}"


def _ref_mn(x: torch.Tensor, M: int) -> torch.Tensor:
    """Independent H_M tensor product Sylvester H_N reference in fp32."""
    out = special_hadamard_reference(x, M)
    if cfg.TO_CPU:
        out = out.cpu()
    return out


@pytest.mark.fast_hadamard_transform_12N
@pytest.mark.fast_hadamard_transform_20N
@pytest.mark.fast_hadamard_transform_28N
@pytest.mark.fast_hadamard_transform_40N
@_skip_if_join_bug
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim,M,tag", HADAMARD_MN_CASES)
@pytest.mark.parametrize("batch", [1, 16, 1024])
def test_fast_hadamard_transform_mn(batch, dim, M, tag, dtype):
    x = torch.randn(batch, dim, dtype=dtype, device=flag_gems.device)
    ref_out = _ref_mn(x, M)
    res_out = _FN_MAP[tag](x)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=dim)


@pytest.mark.fast_hadamard_transform_12N
@_skip_if_join_bug
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_fast_hadamard_transform_mn_scale(scale, dtype):
    batch, dim, M = 16, 6144, 12
    x = torch.randn(batch, dim, dtype=dtype, device=flag_gems.device)
    ref_out = _ref_mn(x, M) * scale
    res_out = flag_gems.fast_hadamard_transform_12N(x, scale=scale)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=dim)


@pytest.mark.fast_hadamard_transform_12N
@_skip_if_join_bug
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 8, 3072), (2, 3, 4, 1536)])
def test_fast_hadamard_transform_mn_leading_dims(shape, dtype):
    x = torch.randn(*shape, dtype=dtype, device=flag_gems.device)
    ref_out = _ref_mn(x, M=12)
    res_out = flag_gems.fast_hadamard_transform_12N(x)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[-1])


@pytest.mark.fast_hadamard_transform_12N
@pytest.mark.fast_hadamard_transform_20N
@pytest.mark.fast_hadamard_transform_28N
@pytest.mark.fast_hadamard_transform_40N
@_skip_if_join_bug
@pytest.mark.parametrize("M", [12, 20, 28, 40])
@pytest.mark.parametrize("dim_case", ["8M", "M+1", "3M"])
def test_fast_hadamard_transform_mn_backward(M, dim_case):
    dim = {"8M": M * 8, "M+1": M + 1, "3M": M * 3}[dim_case]
    scale = -0.5
    x = torch.randn(
        2, dim, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    out = _FN_MAP[f"{M}N"](x, scale=scale)
    ref_out = special_hadamard_reference(utils.to_reference(x.detach()), M, scale=scale)
    utils.gems_assert_close(out, ref_out, torch.float32, reduce_dim=dim)

    grad = torch.randn_like(out)
    (dx,) = torch.autograd.grad(out, x, grad)
    ref_dx = special_hadamard_reference(
        utils.to_reference(grad), M, scale=scale, transpose=True
    )
    utils.gems_assert_close(dx, ref_dx, torch.float32, reduce_dim=dim)
