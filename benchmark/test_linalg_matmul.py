import pytest
import torch

from . import base, consts


def linalg_matmul_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    # linalg_matmul handles both 2D and 3D cases
    # For 2D: (m, k) @ (k, n) -> (m, n), b is unused
    # For 3D: (b, m, k) @ (b, k, n) -> (b, m, n)
    if b == 1:
        # 2D case
        inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2
    else:
        # 3D case (batched)
        inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=device)
        inp2 = torch.randn([b, k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


@pytest.mark.linalg_matmul
def test_linalg_matmul():
    bench = base.BlasBenchmark(
        op_name="linalg_matmul",
        input_fn=linalg_matmul_input_fn,
        torch_op=torch.linalg.matmul,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
