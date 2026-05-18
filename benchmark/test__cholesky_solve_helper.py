import pytest
import torch

from . import base, consts


def cholesky_solve_input_fn(shape, cur_dtype, device):
    n = shape[-1]
    batch_shape = shape[:-2]
    A = torch.randn(*batch_shape, n, n, dtype=cur_dtype, device=device)
    A = A @ A.transpose(-2, -1) + n * torch.eye(n, dtype=cur_dtype, device=device)
    L = torch.linalg.cholesky(A)
    b = torch.randn(*batch_shape, n, 1, dtype=cur_dtype, device=device)
    yield (b, L, False)


class CholeskySolveHelperBenchmark(base.OperationBenchmark):
    def get_input_iter(self, dtype):
        shapes = [(4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (4, 8, 8)]
        for shape in shapes:
            yield from cholesky_solve_input_fn(shape, dtype, self.device)


@pytest.mark.cholesky_solve_helper
def test__cholesky_solve_helper():
    bench = CholeskySolveHelperBenchmark(
        op_name="_cholesky_solve_helper",
        torch_op=torch._cholesky_solve_helper,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
