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

from . import base, consts

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

# ============================================================
# Standard FHT benchmark (hadamard_transform)
# ============================================================

_FHT_SHAPES = [
    (1024, 256),
    (1024, 512),
    (1024, 1024),
    (1024, 4096),
    # (1024, 16384),  # scipy full matrix too slow; temporarily commented
    # (1024, 32768),
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


def _hadamard_transform_ref(x, scale=1.0):
    """Reference matching tests/test_hadamard_transform.py (Dao scipy matrix multiply)."""
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


def torch_ht(x):
    """Benchmark baseline: same scipy Hadamard ref as correctness tests."""
    return _hadamard_transform_ref(x)


class HadamardBenchmark(base.GenericBenchmark2DOnly):
    DEFAULT_SHAPES = _FHT_SHAPES
    DEFAULT_SHAPE_DESC = "batch, dim"

    def set_more_shapes(self):
        return []

    def set_shapes(self, *args, **kwargs):
        # Force _FHT_SHAPES; do not fall back to GenericBenchmark2DOnly in core_shapes.yaml
        self.shapes = self.DEFAULT_SHAPES


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
    """Benchmark baseline: pad to next power of 2, then same scipy ref (not gems)."""
    dim = x.shape[-1]
    padded = 1 << (dim - 1).bit_length()
    x_padded = F.pad(x, (0, padded - dim))
    return _hadamard_transform_ref(x_padded)[..., :dim]


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
