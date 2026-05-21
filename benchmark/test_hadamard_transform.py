import pytest
import torch
import triton
from packaging.version import Version

import flag_gems

from . import base, consts

_TRITON_VERSION = Version(triton.__version__.split("+")[0])
_SKIP_JOIN_BUG = _TRITON_VERSION < Version("3.5.0")
_skip_if_join_bug = pytest.mark.skipif(
    _SKIP_JOIN_BUG,
    reason=f"triton {triton.__version__} has tt.join layout bug (fixed in 3.5.0)",
)

# ============================================================
# Standard FHT benchmark (hadamard_transform)
# ============================================================

_FHT_SHAPES = [
    (1024, 256),
    (1024, 512),
    (1024, 1024),
    (1024, 4096),
    (1024, 16384),
    (1024, 32768),
    (8192, 256),
    (8192, 512),
    (8192, 1024),
    (8192, 4096),
    (8192, 16384),
    (32768, 256),
    (32768, 512),
    (32768, 1024),
    (32768, 4096),
]


def ht_input_fn(shape, dtype, device):
    batch, dim = shape
    yield (torch.randn(batch, dim, dtype=dtype, device=device),)


def _hadamard_matrix(n: int, device) -> torch.Tensor:
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H


def torch_ht(x):
    """Reference: Hadamard matrix multiply in fp32."""
    dim = x.shape[-1]
    padded = 1 << (dim - 1).bit_length() if dim > 1 else 1
    H = _hadamard_matrix(padded, x.device).to(x.dtype)
    x_padded = torch.nn.functional.pad(x, (0, padded - dim))
    return (x_padded @ H.T)[..., :dim]


class HadamardBenchmark(base.GenericBenchmark2DOnly):
    DEFAULT_SHAPES = _FHT_SHAPES
    DEFAULT_SHAPE_DESC = "batch, dim"

    def set_more_shapes(self):
        return []


@pytest.mark.hadamard_transform
@_skip_if_join_bug
def test_hadamard_transform():
    bench = HadamardBenchmark(
        input_fn=ht_input_fn,
        op_name="hadamard_transform",
        torch_op=torch_ht,
        gems_op=flag_gems.hadamard_transform,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# ============================================================
# M×N fused kernel benchmark (hadamard_transform_12N/20N/28N/40N)
# ============================================================

_HT_MN_SHAPES = [
    (1024, 1536),
    (1024, 3072),
    (1024, 6144),
    (1024, 12288),
    (8192, 1536),
    (8192, 3072),
    (8192, 6144),
    (8192, 12288),
    (32768, 1536),
    (32768, 3072),
    (32768, 6144),
    (32768, 12288),
    (1024, 10240),
    (1024, 14336),
    (1024, 20480),
    (8192, 10240),
    (8192, 14336),
    (8192, 20480),
    (1024, 40960),
    (8192, 40960),
]

_TAG_FOR_DIM = {
    1536: "12N",
    3072: "12N",
    6144: "12N",
    12288: "12N",
    10240: "20N",
    20480: "20N",
    14336: "28N",
    40960: "40N",
}

_FN_MAP = {
    "12N": flag_gems.hadamard_transform_12N,
    "20N": flag_gems.hadamard_transform_20N,
    "28N": flag_gems.hadamard_transform_28N,
    "40N": flag_gems.hadamard_transform_40N,
}


def ht_mn_input_fn(shape, dtype, device):
    batch, dim = shape
    yield (torch.randn(batch, dim, dtype=dtype, device=device),)


def torch_ht_mn(x):
    """Reference: pad to next power of 2 and run standard FHT."""
    dim = x.shape[-1]
    padded = 1 << (dim - 1).bit_length()
    x_padded = torch.nn.functional.pad(x, (0, padded - dim))
    return flag_gems.hadamard_transform(x_padded)[..., :dim]


def gems_ht_mn(x):
    dim = x.shape[-1]
    tag = _TAG_FOR_DIM[dim]
    return _FN_MAP[tag](x)


class HadamardMNBenchmark(base.GenericBenchmark2DOnly):
    DEFAULT_SHAPES = _HT_MN_SHAPES
    DEFAULT_SHAPE_DESC = "batch, dim"

    def set_more_shapes(self):
        return []

    def set_shapes(self, *args, **kwargs):
        self.shapes = self.DEFAULT_SHAPES


@pytest.mark.hadamard_transform_mn
@_skip_if_join_bug
def test_hadamard_transform_mn():
    bench = HadamardMNBenchmark(
        input_fn=ht_mn_input_fn,
        op_name="hadamard_transform_mn",
        torch_op=torch_ht_mn,
        gems_op=gems_ht_mn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
