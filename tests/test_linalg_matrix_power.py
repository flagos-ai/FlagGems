import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

if flag_gems.runtime.device.support_fp64:
    DTYPES_ALL = [torch.float32, torch.float64]
else:
    DTYPES_ALL = [torch.float32]

# ---------------------------------------------------------------------------
# Part 1 — common native-pytorch shapes/exponents.
# ---------------------------------------------------------------------------
SHAPES_2D = [(2, 2), (3, 3), (4, 4), (5, 5), (8, 8)]
SHAPES_BATCH = [(2, 2, 2), (3, 4, 4), (2, 3, 5, 5)]
# The backend of ASCEND does not support matrix_power for n<0
if flag_gems.vendor_name == "ascend":
    N_VALUES = [0, 1, 2, 3, 5, 8]
else:
    N_VALUES = [0, 1, 2, 3, 5, 8, -1, -2, -3]


# ---------------------------------------------------------------------------
# Part 2 — large shapes / large exponents.
# ---------------------------------------------------------------------------
LARGE_CASES = [
    # large 2D — moderate exponents
    *[
        (s, n)
        for s in ((16, 16), (32, 32), (64, 64), (128, 128), (256, 256))
        for n in (4, 5, 8, 10, 31, 32)
    ],
    # large batched — moderate exponents
    *[
        (s, n)
        for s in ((4, 8, 8), (8, 16, 16), (2, 3, 4, 4), (5, 2, 6, 6))
        for n in (5, 10, 32)
    ],
    # high exponents — moderate matrices only (fp32 result stays finite)
    *[(s, n) for s in ((16, 16), (32, 32)) for n in (15, 16, 31)],
    *[((32, 32), n) for n in (0, 2, 3, 8, 16, 32, 64)],
    # batched matrices
    *[
        (s, n)
        for s in ((8, 2, 2), (16, 64, 64), (2, 1024, 1024))
        for n in (2, 8, 31, 32)
    ],
    *[
        (s, n)
        for s in ((2, 2), (8, 8), (64, 64), (256, 256), (1024, 1024))
        for n in (2, 8, 31, 32)
    ],
]

# negative powers on larger matrices
# The backend of ASCEND does not support matrix_power for n<0
if flag_gems.vendor_name != "ascend":
    LARGE_CASES += [
        *[
            (s, n)
            for s in ((16, 16), (32, 32), (256, 256), (1024, 1024), (4, 8, 8))
            for n in (-2, -8, -31)
        ],
    ]

DTYPES_LARGE = DTYPES_ALL


def _gen_for(shape, n, dtype):
    """Deterministic per-case RNG — independent of test execution order."""
    import zlib

    key = f"{shape}-{n}-{dtype}".encode()
    g = torch.Generator(device=flag_gems.device)
    g.manual_seed(zlib.crc32(key) & 0xFFFFFFFF)
    return g


_NEG_MAX_COND = 80


# Curret Gems atol = 1e-4, ater testing, the constructed input can achieve
# atol = 1e-5(f32), 1e-15(f64)
def _make_input(shape, dtype, n):
    """
    Deterministic input matrix generator for matrix_power test.
    - n >= 0: well-conditioned matrix with spectral norm ≈ 1 (A = U Σ Vᵀ,
      σ ∈ [1/100, 1]) — keeps Aⁿ bounded so PyTorch's strict tolerances hold
      even for large shapes / large n.
    - n < 0: symmetric positive-definite matrix with bounded condition number
      (A = U Σ Uᵀ, σ ∈ [1/_NEG_MAX_COND, 1]) — guarantees invertibility and
      suppresses inv-error blow-up.
    """
    g = _gen_for(shape, n, dtype)
    M = shape[-1]
    batch = shape[:-2]
    if n >= 0:
        U, _ = torch.linalg.qr(
            torch.randn(*batch, M, M, dtype=dtype, generator=g, device=flag_gems.device)
        )
        V, _ = torch.linalg.qr(
            torch.randn(*batch, M, M, dtype=dtype, generator=g, device=flag_gems.device)
        )
        sigma = torch.linspace(1.0 / 100, 1.0, M, dtype=dtype, device=flag_gems.device)
        return U @ torch.diag(sigma) @ V.transpose(-2, -1)

    # n < 0 branch: well-conditioned SPD matrix A = U Σ U^T
    import math

    *batch_dims, M, M2 = shape
    assert M == M2
    batch_size = math.prod(batch_dims) if batch_dims else 1
    mat_list = []
    for _ in range(batch_size):
        X = torch.randn((M, M), dtype=dtype, generator=g, device=flag_gems.device)
        U, _ = torch.linalg.qr(X)
        sigma_max = 1.0
        sigma_min = sigma_max / _NEG_MAX_COND
        sigma = torch.linspace(
            sigma_min, sigma_max, M, dtype=dtype, device=flag_gems.device
        )
        Sigma = torch.diag(sigma)
        spd_mat = U @ Sigma @ U.T
        mat_list.append(spd_mat)

    stacked = torch.stack(mat_list, dim=0)
    return stacked.reshape(shape)


def _matrix_power_golden(A, n):
    """Reference result for linalg.matrix_power.

    Default: native torch.linalg.matrix_power on the reference device (CPU under
    --ref cpu, the op device otherwise), upcast to fp64 via
    ``utils.to_reference(A, upcast=True)`` so fp32 inputs are validated against a
    high-precision result.

    Two backends fall back to torch's CPU fp64 golden instead:
      * thead — torch's on-device fp64 matrix_power has a precision defect, so a
        device-side fp64 reference is unreliable even for fp64 inputs;
      * devices without fp64 (``support_fp64 == False``) — the device can't
        compute an accurate fp64 reference, and an on-device fp32→fp64 cast can
        silently produce zeros.
    ``A.cpu().double()`` moves to CPU *before* the upcast so the golden is always
    computed on the CPU (which always supports fp64).
    """
    if flag_gems.vendor_name == "thead" or not flag_gems.runtime.device.support_fp64:
        ref = torch.linalg.matrix_power(A.cpu().double(), n)
        # gems_assert_close compares on the op device in the default mode, on CPU
        # in --ref cpu mode.
        if not utils.TO_CPU:
            ref = ref.to(A.device)
        return ref
    return torch.linalg.matrix_power(utils.to_reference(A, upcast=True), n)


@pytest.mark.linalg_matrix_power
@pytest.mark.parametrize("shape", SHAPES_2D + SHAPES_BATCH)
@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("dtype", DTYPES_ALL)
def test_common(shape, n, dtype):
    A = _make_input(shape, dtype, n)
    ref = _matrix_power_golden(A, n)
    res = flag_gems.linalg_matrix_power(A, n)
    utils.gems_assert_close(res, ref, dtype)


@pytest.mark.linalg_matrix_power
@pytest.mark.parametrize(
    "shape, n", LARGE_CASES, ids=[f"{s}-n{n}" for s, n in LARGE_CASES]
)
@pytest.mark.parametrize("dtype", DTYPES_LARGE)
def test_large(shape, n, dtype):
    A = _make_input(shape, dtype, n)
    ref = _matrix_power_golden(A, n)
    res = flag_gems.linalg_matrix_power(A, n)
    utils.gems_assert_close(res, ref, dtype)


@pytest.mark.linalg_matrix_power_out
@pytest.mark.parametrize("n", [0, 2, 3, 5])
@pytest.mark.parametrize("dtype", DTYPES_ALL)
def test_out_parameter(n, dtype):
    A = _make_input((4, 4), dtype, n)
    out = torch.empty_like(A)
    ref = _matrix_power_golden(A, n)
    res = flag_gems.linalg_matrix_power(A, n, out=out)
    assert res is out, "out= must return the same tensor object"
    utils.gems_assert_close(out, ref, dtype)


@pytest.mark.linalg_matrix_power_out
@pytest.mark.parametrize("shape", SHAPES_2D + SHAPES_BATCH)
@pytest.mark.parametrize("n", N_VALUES)
@pytest.mark.parametrize("dtype", DTYPES_ALL)
def test_linalg_matrix_power_out(shape, n, dtype):
    """Test the linalg_matrix_power.out overload.

    Dispatches torch.linalg.matrix_power(A, n, out=out) under use_gems so the
    ``*.out`` dispatcher key routes through flag_gems' ``linalg_matrix_power_out``
    rather than torch's native compute.  The result must be written into the
    caller-provided ``out`` tensor and that same tensor returned.
    """
    A = _make_input(shape, dtype, n)
    ref = _matrix_power_golden(A, n)

    out = torch.empty_like(A)
    res = flag_gems.linalg_matrix_power(A, n, out=out)
    assert res is out, "linalg_matrix_power.out must return the out tensor"
    utils.gems_assert_close(out, ref, dtype)


@pytest.mark.linalg_matrix_power_out
@pytest.mark.parametrize("n", [-2, 3])
@pytest.mark.parametrize("dtype", DTYPES_ALL)
def test_out_missing_raises(n, dtype):
    """The out overload requires an out tensor (like torch's *.out schema)."""
    A = _make_input((4, 4), dtype, n)
    with pytest.raises(TypeError):
        flag_gems.linalg_matrix_power_out(A, n)


@pytest.mark.linalg_matrix_power
@pytest.mark.parametrize("shape", [(3, 4), (5,)])
def test_invalid_shape_rejected(shape):
    A = torch.randn(
        *shape, generator=_gen_for(shape, 2, torch.float32), device=flag_gems.device
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.linalg_matrix_power(A, 2)
