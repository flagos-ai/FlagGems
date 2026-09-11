import pytest
import torch

import flag_gems

from . import base, utils

VENDOR = flag_gems.vendor_name

DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64:
    DTYPES.append(torch.float64)

SHAPE = [
    (1024,),
    (65536,),
    (64, 64),
    (256, 512),
    (2, 2048),
    (4, 32, 64),
    (8, 4, 32, 32),
    (1024, 1024),
]


# Vector norm orders (1D inputs, dim=None flattens).
_VECTOR_ORDS = (2, 1, 0, float("inf"))

# Matrix norm orders (2D inputs, dim=None → (-2, -1)).
_MATRIX_ORDS = ("fro", 1, -1, float("inf"), float("-inf"))

# Non-default dims exercised on batched inputs.
_BATCH_DIMS = [(-2, -1), (0, 2), 1]


def _svd_ok(shape):
    """GEMS SVD kernel limits: k = min(m, n) ∈ [2, 512], max(m, n) ≤ 2048."""
    k, rows = min(shape[-2], shape[-1]), max(shape[-2], shape[-1])
    return k >= 2 and k <= 512 and rows <= 2048


class LinalgNormBenchmark(base.Benchmark):
    def get_input_iter(self, dtype):
        for shape in SHAPE:
            if dtype == torch.float64:
                inp = torch.randn(shape, dtype=dtype, device=self.device)
            else:
                inp = utils.generate_tensor_input(shape, dtype, self.device)

            if len(shape) == 1:
                # vector norm: dim=None (flatten)
                for ord_val in _VECTOR_ORDS:
                    yield inp.clone(), ord_val
            elif len(shape) == 2:
                # matrix norm: dim=None defaults to (-2, -1)
                if dtype in (torch.float16, torch.bfloat16):
                    # torch's reference routes these to linalg_matrix_norm, which
                    # rejects low-precision dtypes -- no baseline to compare against.
                    return
                for ord_val in _MATRIX_ORDS:
                    yield inp.clone(), ord_val
                if _svd_ok(shape):
                    # Nuclear norm (SVD-based); GEMS kernel caps k ∈ [2, 512].
                    yield inp.clone(), "nuc"
            else:
                # batched: dim=None (matrix, (-2, -1)) + explicit 2-tuple/int dims
                if dtype not in (torch.float16, torch.bfloat16):
                    yield inp.clone(), None
                # ord=2 with a 2-tuple dim is the spectral norm (SVD) -- torch rejects
                # fp16/bf16 there, so only benchmark it on fp32/fp64.
                for ord_val, dim in ((1, _BATCH_DIMS[1]), (2, _BATCH_DIMS[2])):
                    yield inp.clone(), ord_val, dim
                if dtype not in (torch.float16, torch.bfloat16):
                    yield inp.clone(), 2, _BATCH_DIMS[0]
                    if _svd_ok(shape):
                        yield inp.clone(), "nuc", _BATCH_DIMS[0]


@pytest.mark.linalg_norm
def test_linalg_norm():
    bench = LinalgNormBenchmark(
        op_name="linalg_norm",
        torch_op=torch.linalg.norm,
        gems_op=flag_gems.linalg_norm,
        dtypes=DTYPES,
    )
    bench.run()
