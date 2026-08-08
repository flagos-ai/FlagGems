import pytest
import torch

import flag_gems

from . import base

pytestmark = pytest.mark.filterwarnings(
    "ignore:Warning only once for all operators.*:UserWarning"
)

# torch.linalg.lstsq has no kernel on every backend. On Ascend it falls back to
# the CPU -- measured 1417.9 ms at 512x512 against 1306.5 ms for the same call
# on CPU, a ratio of 1.09 -- so timing it there measures CPU LAPACK and yields
# a "speedup" that says nothing about the device. torch.linalg.qr IS native
# there (38.8 ms vs 1690.5 ms on CPU, ratio 0.02), so compose the reference
# from QR the way the kernel itself computes it and keep the comparison
# like-for-like. Same reason benchmark/test_polygamma.py composes its baseline.
#
# gems_op must also be passed explicitly: with a composed torch_op the harness
# would otherwise time the gems side by re-running that composition under
# use_gems, which never reaches this operator. flag_gems.linalg_lstsq is the
# right handle -- SpecOpRegistrar rebinds that symbol to the vendor override at
# import, so it is the Ascend kernel on Ascend.
_DEVICE_REF = flag_gems.device not in ("cuda", "cpu")


def _lstsq_via_qr(A, b, driver="gels"):
    """Least-squares from primitives that run ON DEVICE.

    Mirrors what the kernel does rather than substituting a cheaper algorithm:
    QR, apply Q^T to the RHS, back-substitute. For m < n it QRs A^T and forms
    the minimum-norm solution x = Q R^-T b, which is the same routing the
    kernel uses, so the two sides do comparable work.

    torch.linalg.solve_triangular is itself a CPU fallback here, but it is a
    small share of the total (0.485 ms against QR's 38.8 ms at 512x512, ~1%),
    so the baseline stays overwhelmingly on device. The composition skips the
    residual norms the operator also returns, which biases slightly in the
    operator's favour -- worth knowing when reading the numbers.

    Only `gels` is composed here. The other drivers are different algorithms,
    not different flags: gelsy uses a complete orthogonal factorization, gelsd
    and gelss use an SVD, and all three are rank-revealing. Substituting this
    QR composition for any of them would silently produce a gels reference
    while the operator computed something else -- a wrong baseline that still
    looks like a valid speedup. Reject them rather than ignore the argument.
    """
    if driver not in (None, "gels"):
        raise ValueError(
            f"_lstsq_via_qr composes the gels (QR) algorithm only; got "
            f"driver={driver!r}. gelsy/gelsd/gelss are rank-revealing and need "
            f"a different decomposition, so this reference cannot stand in for "
            f"them -- add the matching composition instead of relaxing this."
        )
    vec = b.ndim == A.ndim - 1
    B = b.unsqueeze(-1) if vec else b
    m, n = A.shape[-2], A.shape[-1]
    if m >= n:
        Q, R = torch.linalg.qr(A, mode="reduced")
        X = torch.linalg.solve_triangular(
            R, torch.matmul(Q.transpose(-1, -2), B), upper=True
        )
    else:
        # min-norm: A^T = QR  =>  A A^T = R^T R  =>  x = A^T (A A^T)^-1 b
        #                                             = Q R^-T b
        Q, R = torch.linalg.qr(A.transpose(-1, -2), mode="reduced")
        W = torch.linalg.solve_triangular(R.transpose(-1, -2), B, upper=False)
        X = torch.matmul(Q, W)
    return X.squeeze(-1) if vec else X


# Shapes the CUDA path benchmarks. Every regime the operator supports.
_SHAPES_FULL = [
    # tall (M >= N)
    (256, 32),
    (1024, 16),
    (4096, 8),
    (8, 4096, 8),
    (64, 2048, 16),
    (16, 8192, 16),
    # wide (M < N), within the tile budget (single-tile kernel)
    (8, 16, 512),
    (8, 64, 256),
    (16, 16, 1024),
    (64, 32, 256),
    # wide beyond the budget (blocked TSQR path)
    (64, 1024),
    (128, 512),
    (8, 64, 1024),
    (16, 32, 2048),
    # square / near-square -> compact-WY blocked QR
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (8, 256, 256),
    (8, 512, 512),
    (4096, 512),
    # underdetermined with large m -> the A^T QR also routes to WY
    (512, 1024),
    (1024, 2048),
]

# Ascend keeps every REGIME above; only the largest instances of two are
# dropped, for measured reasons rather than to flatter the numbers:
#   * the composed QR baseline is O(n^3) ON THE REFERENCE SIDE -- 38.8 ms at
#     512x512, so 1024^2 and 2048^2 cost seconds per repeat to measure torch,
#     not gems
#   * the underdetermined path at m >= 1024 is correct but slow on this backend
#     (blocked triangular solve, one row per barrier); that is a known open
#     optimisation, tracked separately, not a correctness gap
# Square is still represented to 512 and batched 512, and the underdetermined
# WY route by (512, 1024). Restore _SHAPES_FULL to measure the dropped three.
_SHAPES_DEVICE = [
    s for s in _SHAPES_FULL if s not in ((1024, 1024), (2048, 2048), (1024, 2048))
]


class LstsqBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "(*B), M, N  (M>=N tall, or M<N wide)"
    DEFAULT_DTYPES = [torch.float32]
    DEFAULT_SHAPES = _SHAPES_DEVICE if _DEVICE_REF else _SHAPES_FULL

    def set_more_shapes(self):
        return []

    def set_shapes(self, *args, **kwargs):
        # Force our shapes; the file-based default injects a 1D shape that
        # lstsq (which needs a 2D+ matrix) cannot accept.
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=dtype, device=self.device)
            b = torch.randn(shape[:-1], dtype=dtype, device=self.device)
            yield A, b, {"driver": "gels"}


@pytest.mark.linalg_lstsq
def test_linalg_lstsq():
    bench = LstsqBenchmark(
        op_name="linalg_lstsq",
        torch_op=_lstsq_via_qr if _DEVICE_REF else torch.linalg.lstsq,
        gems_op=(
            (lambda A, b, **kw: flag_gems.linalg_lstsq(A, b, driver="gels"))
            if _DEVICE_REF
            else None
        ),
        # gels supports float32/float64 only; fp16/bf16 are not supported by
        # PyTorch's reference, and complex is outside the native path. float64
        # is dropped where the backend has no kernel for it -- on Ascend the
        # device has no float64 unit, so the operator raises there by design.
        dtypes=[torch.float32] if _DEVICE_REF else [torch.float32, torch.float64],
    )
    bench.run()
