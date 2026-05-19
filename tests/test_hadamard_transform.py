import pytest
import torch
import triton
from packaging.version import Version

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

_TRITON_VERSION = Version(triton.__version__.split("+")[0])
_SKIP_JOIN_BUG = _TRITON_VERSION < Version("3.5.0")
_skip_if_join_bug = pytest.mark.skipif(
    _SKIP_JOIN_BUG,
    reason=f"triton {triton.__version__} has tt.join layout bug (fixed in 3.5.0)",
)

if cfg.QUICK_MODE:
    HADAMARD_MN_CASES = [(1536, 3, "12N"), (10240, 5, "20N"), (14336, 7, "28N")]
else:
    HADAMARD_MN_CASES = [
        (1536, 3, "12N"),
        (3072, 3, "12N"),
        (6144, 3, "12N"),
        (12288, 3, "12N"),
        (10240, 5, "20N"),
        (20480, 5, "20N"),
        (14336, 7, "28N"),
        (20480, 5, "40N"),
        (40960, 5, "40N"),
    ]

_FN_MAP = {
    3: flag_gems.hadamard_transform_12N,
    5: flag_gems.hadamard_transform_20N,
    7: flag_gems.hadamard_transform_28N,
}


def _ref_mn(x: torch.Tensor, M: int) -> torch.Tensor:
    """Reference: 2-kernel version (H_M column transform in fp32 + standard FHT)."""
    *leading, dim = x.shape
    batch = x.numel() // dim
    n_cols = dim // M
    orig_dtype = x.dtype
    xm = x.reshape(batch, M, n_cols).float()

    if M == 3:
        a, b, c = xm[:, 0], xm[:, 1], xm[:, 2]
        rows = [a + b + c, a - b + c, a + b - c]
    elif M == 5:
        a, b, c, d, e = xm[:, 0], xm[:, 1], xm[:, 2], xm[:, 3], xm[:, 4]
        rows = [
            a + b + c + d + e,
            a - b + c - d + e,
            a + b - c + d - e,
            a - b - c - d - e,
            a + b + c - d - e,
        ]
    elif M == 7:
        a, b, c, d, e, f, g = (xm[:, i] for i in range(7))
        rows = [
            a + b + c + d + e + f + g,
            a - b + c - d + e - f + g,
            a + b - c + d - e + f - g,
            a - b - c - d - e - f - g,
            a + b + c - d - e - f - g,
            a - b + c + d - e + f + g,
            a + b - c - d + e + f - g,
        ]
    else:
        raise ValueError(f"Unsupported M={M}")

    ym = torch.stack(rows, dim=1).reshape(batch * M, n_cols)  # keep fp32
    ym = flag_gems.hadamard_transform(ym)  # FHT in fp32
    return ym.to(orig_dtype).reshape(*leading, dim)


@pytest.mark.hadamard_transform_mn
@_skip_if_join_bug
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim,M,tag", HADAMARD_MN_CASES)
@pytest.mark.parametrize("batch", [1, 16, 1024])
def test_hadamard_transform_mn(batch, dim, M, tag, dtype):
    x = torch.randn(batch, dim, dtype=dtype, device=flag_gems.device)
    ref_out = _ref_mn(x, M)
    res_out = _FN_MAP[M](x)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=dim)


@pytest.mark.hadamard_transform_mn
@_skip_if_join_bug
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_hadamard_transform_mn_scale(scale, dtype):
    batch, dim, M = 16, 6144, 3
    x = torch.randn(batch, dim, dtype=dtype, device=flag_gems.device)
    ref_out = _ref_mn(x, M) * scale
    res_out = flag_gems.hadamard_transform_12N(x, scale=scale)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=dim)


@pytest.mark.hadamard_transform_mn
@_skip_if_join_bug
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 8, 3072), (2, 3, 4, 1536)])
def test_hadamard_transform_mn_leading_dims(shape, dtype):
    x = torch.randn(*shape, dtype=dtype, device=flag_gems.device)
    ref_out = _ref_mn(x, M=3)
    res_out = flag_gems.hadamard_transform_12N(x)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[-1])
