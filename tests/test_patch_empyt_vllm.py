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

"""
Unit tests for patch_empty_vllm() function.

Tests the registration of custom operators to their corresponding torch.ops namespaces.
"""

import pytest
import torch

import flag_gems
from flag_gems.patches import patch_empty_vllm


@pytest.mark.patch_empty_vllm
def test_registers_operators_to_torch_ops_namespace():
    """After patch_empty_vllm(), operators should be accessible via torch.ops._C etc."""
    patch_empty_vllm()

    # _C operators
    assert hasattr(torch.ops, "_C")
    assert hasattr(torch.ops._C, "silu_and_mul")
    assert hasattr(torch.ops._C, "silu_and_mul_with_clamp")
    assert hasattr(torch.ops._C, "rms_norm")
    assert hasattr(torch.ops._C, "cutlass_scaled_mm")

    # _moe_C operators
    assert hasattr(torch.ops, "_moe_C")
    assert hasattr(torch.ops._moe_C, "topk_softmax")
    assert hasattr(torch.ops._moe_C, "moe_align_block_size")


@pytest.mark.patch_empty_vllm
def test_idempotent_multiple_calls():
    """Calling patch_empty_vllm() multiple times should not raise errors."""
    patch_empty_vllm()
    patch_empty_vllm()  # Should not raise
    patch_empty_vllm()  # Should not raise

    assert hasattr(torch.ops._C, "silu_and_mul")


@pytest.mark.patch_empty_vllm
def test_silu_and_mul_callable():
    """torch.ops._C.silu_and_mul should be callable with tensors."""
    patch_empty_vllm()

    # _C.silu_and_mul has signature: (Tensor(a!) out, Tensor input) -> ()
    inp = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    out = torch.empty(4, 8, dtype=torch.float16, device=flag_gems.device)

    torch.ops._C.silu_and_mul(out, inp)

    assert out.shape == (4, 8)
    assert out.dtype == inp.dtype


@pytest.mark.patch_empty_vllm
def test_silu_and_mul_with_clamp_callable():
    """torch.ops._C.silu_and_mul_with_clamp should accept limit parameter."""
    patch_empty_vllm()

    # _C.silu_and_mul_with_clamp has signature: (Tensor(a!) out, Tensor input, float limit) -> ()
    inp = torch.randn(4, 16, dtype=torch.float16, device=flag_gems.device)
    out = torch.empty(4, 8, dtype=torch.float16, device=flag_gems.device)
    limit = 1.0

    torch.ops._C.silu_and_mul_with_clamp(out, inp, limit)

    assert out.shape == (4, 8)
    assert out.dtype == inp.dtype


@pytest.mark.patch_empty_vllm
def test_rms_norm_callable():
    """torch.ops._C.rms_norm should be callable."""
    patch_empty_vllm()

    hidden_size = 64
    inp = torch.randn(4, hidden_size, dtype=torch.float16, device=flag_gems.device)
    weight = torch.ones(hidden_size, dtype=torch.float16, device=flag_gems.device)
    result = torch.empty_like(inp)
    epsilon = 1e-5

    torch.ops._C.rms_norm(result, inp, weight, epsilon)

    assert result.shape == inp.shape
    assert result.dtype == inp.dtype


@pytest.mark.patch_empty_vllm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_and_mul_dtypes(dtype):
    """torch.ops._C.silu_and_mul should work with different dtypes."""
    patch_empty_vllm()

    inp = torch.randn(4, 16, dtype=dtype, device=flag_gems.device)
    out = torch.empty(4, 8, dtype=dtype, device=flag_gems.device)

    torch.ops._C.silu_and_mul(out, inp)

    assert out.dtype == dtype


@pytest.mark.patch_empty_vllm
@pytest.mark.parametrize("shape", [(4, 16), (8, 32), (2, 64)])
def test_silu_and_mul_shapes(shape):
    """torch.ops._C.silu_and_mul should work with different shapes."""
    patch_empty_vllm()

    inp = torch.randn(shape, dtype=torch.float16, device=flag_gems.device)
    out_shape = (shape[0], shape[1] // 2)
    out = torch.empty(out_shape, dtype=torch.float16, device=flag_gems.device)

    torch.ops._C.silu_and_mul(out, inp)

    assert out.shape == out_shape
