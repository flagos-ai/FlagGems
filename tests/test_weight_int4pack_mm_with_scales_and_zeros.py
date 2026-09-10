import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# ---------------------------------------------------------------------------
# Reference implementation: full dequantization + matmul in eager Python
# ---------------------------------------------------------------------------


def _reference_dequant_weight(mat2, qScale, qZeros, qGroupSize, K, N):
    """Dequantize packed uint8 int4 weights to float32.

    Convention (identical to the Triton kernel):
      mat2  shape = (N, K//2), dtype=uint8
      Each byte packs two int4 values: low nibble (bits 0-3) for even column 2*j,
      high nibble (bits 4-7) for odd column (2*j+1).
      int4 values are unsigned 0..15.
      dequant: w = (q - zero) * scale
      qZeros / qScale  shape = (K // qGroupSize, N)
      Grouping is along K: features k in [g*qGroupSize, (g+1)*qGroupSize)
      share qScale[g, :] / qZeros[g, :].
    """
    weight_f32 = torch.empty((K, N), dtype=torch.float32, device=mat2.device)
    for k in range(K):
        byte_idx = k // 2
        group = k // qGroupSize
        # extract int4 from the packed byte
        bytes_val = mat2[:, byte_idx].to(torch.int32)
        if k % 2 == 0:
            int4_vals = bytes_val & 0xF
        else:
            int4_vals = (bytes_val >> 4) & 0xF
        # dequant
        scales = qScale[group, :]
        zeros = qZeros[group, :]
        w_k = (int4_vals.to(torch.float32) - zeros) * scales
        weight_f32[k, :] = w_k
    return weight_f32


def reference_mm(A, mat2, qGroupSize, qScale, qZeros):
    """Full reference: dequantize + matmul."""
    M, K = A.shape
    N = mat2.shape[0]
    W_f32 = _reference_dequant_weight(mat2, qScale, qZeros, qGroupSize, K, N)
    return torch.mm(A.to(torch.float32), W_f32).to(A.dtype)


# ---------------------------------------------------------------------------
# Shape and dtype helpers
# ---------------------------------------------------------------------------

# Use small shapes to keep test runtime manageable.
# (M, K, N) where K must be even and K % qGroupSize == 0.
MM_SHAPES = [
    (4, 16, 8),
    (8, 32, 16),
    (4, 64, 16),
    (8, 128, 32),
]


def _make_test_case(M, K, N, qGroupSize, dtype):
    """Create valid int4-packed mat2 and corresponding qScale/qZeros tensors.

    Returns (A, mat2_uint8, qGroupSize, qScale, qZeros) that satisfy all
    the kernel's input conventions.
    """
    device = flag_gems.device

    # Activation
    A = torch.randn((M, K), dtype=dtype, device=device) * 0.1

    # Random int4 weights (0..15) in int32, then pack
    W_int32 = torch.randint(0, 16, (K, N), dtype=torch.int32, device=device)
    K2 = K // 2
    mat2 = torch.empty((N, K2), dtype=torch.uint8, device=device)
    for k in range(K):
        byte_idx = k // 2
        if k % 2 == 0:
            mat2[:, byte_idx] = (W_int32[k, :].to(torch.uint8) & 0xF).to(torch.uint8)
        else:
            mat2[:, byte_idx] = (
                mat2[:, byte_idx].to(torch.int32)
                | ((W_int32[k, :].to(torch.uint8) & 0xF).to(torch.int32) << 4)
            ).to(torch.uint8)

    G = K // qGroupSize
    qScale = torch.randn((G, N), dtype=dtype, device=device).abs() + 0.01
    qZeros = torch.randn((G, N), dtype=dtype, device=device) * 2.0

    return A, mat2, qGroupSize, qScale, qZeros


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.weight_int4pack_mm_with_scales_and_zeros
@pytest.mark.parametrize("M,K,N", MM_SHAPES)
@pytest.mark.parametrize(
    "qGroupSize",
    [p for p in [2, 4, 8, 16] if any(p <= s[1] and s[1] % p == 0 for s in MM_SHAPES)][
        :3
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_weight_int4pack_mm_with_scales_and_zeros(M, K, N, qGroupSize, dtype):
    if K % qGroupSize != 0:
        pytest.skip("K must be divisible by qGroupSize")

    A, mat2, qgs, qScale, qZeros = _make_test_case(M, K, N, qGroupSize, dtype)

    # Move everything to CPU for reference computation
    ref_A = A.clone().to("cpu")
    ref_mat2 = mat2.clone().to("cpu")
    ref_qScale = qScale.clone().to("cpu")
    ref_qZeros = qZeros.clone().to("cpu")

    ref_out = reference_mm(ref_A, ref_mat2, qgs, ref_qScale, ref_qZeros)

    # Get result from GEMS
    with flag_gems.use_gems():
        res_out = torch.ops.aten._weight_int4pack_mm_with_scales_and_zeros(
            A, mat2, qgs, qScale, qZeros
        )

    # Move both to CPU for comparison
    res_out = res_out.to("cpu")
    ref_out = ref_out.to("cpu")
    utils.gems_assert_close(res_out, ref_out, dtype, atol=0.1)
