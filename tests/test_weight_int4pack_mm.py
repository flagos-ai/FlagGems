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

# The reference baseline is the real ATen operator. On this platform the CUDA
# `torch.ops.aten._weight_int4pack_mm` requires NVIDIA's Marlin tiled weight
# format (and matching kernel support) that is not runnable here, so we use the
# packing-compatible CPU overload `_weight_int4pack_mm_for_cpu` as the golden.
# Both share the byte-pair uint8 (N, K//2) weight layout produced by
# `_convert_weight_to_int4pack_for_cpu`, which matches this operator's packing.
_ATEN_CPU_MM = getattr(torch.ops.aten, "_weight_int4pack_mm_for_cpu", None)
_ATEN_CPU_CONVERT = getattr(torch.ops.aten, "_convert_weight_to_int4pack_for_cpu", None)


def _aten_reference_weight_int4pack_mm(A, weight_int4, qGroupSize, scale, zero):
    """Golden output from the real ATen `_weight_int4pack_mm_for_cpu`.

    This operator's dequantization convention is `w = (q - zero) * scale`,
    while ATen dequantizes as `w = scale * (q - 8) + zero_add`. The two are
    identical when `zero_add = scale * (8 - zero)`, so we reparametrize the
    scale/zero pair before handing them to ATen. All ATen work runs on CPU in
    float32 for a precise, dtype-agnostic golden.

    Args:
        A:            activation tensor (M, K).
        weight_int4:  raw int4 weights (N, K), int32, values in 0..15.
        qGroupSize:   quantization group size along K.
        scale, zero:  this operator's per-group scale / zero, shape
                      (K // qGroupSize, N).

    Returns:
        Golden output (M, N) as float32 on CPU.
    """
    A_cpu = A.detach().to(torch.float32).cpu()
    weight_cpu = weight_int4.detach().to(torch.int32).cpu()
    scale_cpu = scale.detach().to(torch.float32).cpu()
    zero_cpu = zero.detach().to(torch.float32).cpu()

    # ATen expects its own tiled byte-pair packing; build it from the same
    # raw int4 weights this operator consumes.
    packed = _ATEN_CPU_CONVERT(weight_cpu, 2)

    # Reparametrize (q - zero) * scale  ->  scale * (q - 8) + zero_add.
    zero_add = scale_cpu * (8.0 - zero_cpu)
    qScaleAndZeros = torch.stack([scale_cpu, zero_add], dim=-1).contiguous()

    golden = _ATEN_CPU_MM(A_cpu, packed, qGroupSize, qScaleAndZeros)

    # Keep the golden on the same device as the gems result unless the test
    # harness compares on CPU (TO_CPU), matching utils.to_reference semantics.
    if not utils.TO_CPU:
        golden = golden.to(A.device)
    return golden


def _create_int4_packed_weights(weight_int4, N, K):
    """Pack int4 weights (int32, shape (N, K)) into uint8 byte-pair format.

    Packing: (odd_val << 4) | even_val  (matches _convert_weight_to_int4pack).
    """
    even = (weight_int4[:, 0::2] & 0xF).to(torch.uint8)
    odd = (weight_int4[:, 1::2] & 0xF).to(torch.uint8)
    return ((odd << 4) | even).contiguous()


def _create_scale_and_zeros(K, N, qGroupSize, dtype, device):
    """Create random scale and zero tensors for testing.

    scales are in range (0.5, 2.0), zeros are in range 4..10 (int4 range 0..15).
    Returns (qScaleAndZeros, scale, zero); qScaleAndZeros stacks scale/zero on
    the last dim as this operator expects.
    """
    num_groups = K // qGroupSize
    scales = torch.rand((num_groups, N), dtype=dtype, device=device) * 1.5 + 0.5
    # zero points: small integer offset from center of int4 range
    zeros = torch.randint(4, 11, (num_groups, N), dtype=dtype, device=device)
    qScaleAndZeros = torch.stack([scales, zeros], dim=-1)
    return qScaleAndZeros, scales, zeros


# N must be divisible by 16 for the ATen CPU int4 packer, and qGroupSize even.
@pytest.mark.skipif(
    _ATEN_CPU_MM is None or _ATEN_CPU_CONVERT is None,
    reason="aten._weight_int4pack_mm_for_cpu not available in this torch build",
)
@pytest.mark.weight_int4pack_mm
@pytest.mark.parametrize(
    "M,N,K,qGroupSize",
    [
        (8, 16, 64, 32),
        (16, 16, 128, 32),
        (16, 32, 128, 64),
        (32, 64, 256, 64),
        (64, 32, 128, 64),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weight_int4pack_mm(M, N, K, qGroupSize, dtype):
    """Test _weight_int4pack_mm accuracy against the real ATen operator."""
    device = flag_gems.device

    # Create activation tensor
    A = torch.randn((M, K), dtype=dtype, device=device)

    # Create int4 weights (values 0..15)
    weight_int4 = torch.randint(0, 16, (N, K), dtype=torch.int32, device=device)

    # Pack weights into byte-pair format
    mat2_packed = _create_int4_packed_weights(weight_int4, N, K)

    # Create scales and zeros
    qScaleAndZeros, scale, zero = _create_scale_and_zeros(
        K, N, qGroupSize, dtype, device
    )

    # Golden reference from the real ATen operator (CPU, float32).
    ref_out = _aten_reference_weight_int4pack_mm(
        A, weight_int4, qGroupSize, scale, zero
    ).to(dtype)

    # GEMS computation
    res_out = flag_gems.weight_int4pack_mm(A, mat2_packed, qGroupSize, qScaleAndZeros)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2, reduce_dim=K)
