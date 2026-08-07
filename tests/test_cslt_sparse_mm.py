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

"""Accuracy tests for the Triton ``_cslt_sparse_mm`` implementation.

The cuSPARSELt compressed format is produced by ``torch._cslt_compress``,
which only runs on CUDA, so the reference is the native CUDA op rather than a
CPU path. The FlagGems Triton implementation decodes the compressed blob and
runs a dense matmul; results are compared against ``torch._cslt_sparse_mm``.
"""

import pytest
import torch

import flag_gems
from flag_gems.ops._cslt_sparse_mm import _cslt_sparse_mm_enabled

from . import accuracy_utils as utils

CSLT_AVAILABLE = (
    torch.cuda.is_available()
    and hasattr(torch, "_cslt_compress")
    and getattr(torch.backends, "cusparselt", None) is not None
    and torch.backends.cusparselt.is_available()
)

# The cuSPARSELt compressed layout is vendor internal and differs per GPU
# architecture, so the Triton decoder is only registered where its swizzle is
# known to be correct. Skip rather than fail elsewhere: on those devices
# use_gems() intentionally leaves the native op in place, so there is nothing
# of ours to compare against.
ARCH_SUPPORTED = CSLT_AVAILABLE and _cslt_sparse_mm_enabled()

pytestmark = [
    pytest.mark.skipif(
        not CSLT_AVAILABLE, reason="cuSPARSELt not available on this device"
    ),
    pytest.mark.skipif(
        CSLT_AVAILABLE and not ARCH_SUPPORTED,
        reason=(
            "the Triton _cslt_sparse_mm decoder models the Hopper cuSPARSELt "
            "metadata layout; not registered on this architecture"
        ),
    ),
]


def _make_2to4(M, K, dtype, device):
    """Build an M x K matrix with an exact 2:4 sparsity pattern."""
    a = torch.randn(M, K, dtype=dtype, device=device).view(M, K // 4, 4)
    # keep the 2 largest-magnitude of every group of 4, zero the other 2.
    idx = a.abs().argsort(dim=-1)
    mask = torch.zeros_like(a)
    mask.scatter_(-1, idx[..., 2:], 1.0)
    return (a * mask).view(M, K).contiguous()


@pytest.mark.cslt_sparse_mm
@pytest.mark.parametrize("shape", [(32, 64, 16), (64, 128, 32), (128, 256, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_cslt_sparse_mm(shape, dtype):
    M, K, N = shape
    dev = flag_gems.device
    A = _make_2to4(M, K, dtype, dev)
    compressed_A = torch._cslt_compress(A)
    B = torch.randn(K, N, dtype=dtype, device=dev)

    ref_out = torch._cslt_sparse_mm(compressed_A, B)
    with flag_gems.use_gems():
        res_out = torch._cslt_sparse_mm(compressed_A, B)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cslt_sparse_mm
@pytest.mark.parametrize("shape", [(64, 128, 32), (128, 256, 64)])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_accuracy_cslt_sparse_mm_with_alpha(shape, dtype):
    M, K, N = shape
    dev = flag_gems.device
    A = _make_2to4(M, K, dtype, dev)
    compressed_A = torch._cslt_compress(A)
    B = torch.randn(K, N, dtype=dtype, device=dev)
    alpha = torch.tensor(2.0, dtype=dtype, device=dev)

    ref_out = torch._cslt_sparse_mm(compressed_A, B, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch._cslt_sparse_mm(compressed_A, B, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.cslt_sparse_mm
@pytest.mark.parametrize("shape", [(64, 128, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_cslt_sparse_mm_transpose(shape, dtype):
    M, K, N = shape
    dev = flag_gems.device
    A = _make_2to4(M, K, dtype, dev)
    compressed_A = torch._cslt_compress(A)
    B = torch.randn(K, N, dtype=dtype, device=dev)

    ref_out = torch._cslt_sparse_mm(compressed_A, B, transpose_result=True)
    with flag_gems.use_gems():
        res_out = torch._cslt_sparse_mm(compressed_A, B, transpose_result=True)

    assert res_out.shape == ref_out.shape
    utils.gems_assert_close(res_out, ref_out, dtype)
