import pytest
import torch

from . import base, consts, utils


def unflatten_input_fn(shape, dtype, device):
    # Unflatten dim 0 into different factorizations
    dim = 0
    dim_size = shape[dim]

    # Find valid factorizations
    factors = []
    for f in range(2, min(dim_size + 1, 17)):
        if dim_size % f == 0:
            factors.append(f)

    if not factors:
        factors = [1]

    for factor in factors[:4]:
        inp = utils.generate_tensor_input(shape, dtype, device)
        sizes = (factor, dim_size // factor)
        yield inp, {
            "dim": dim,
            "sizes": sizes,
        }


@pytest.mark.unflatten
def test_unflatten():
    bench = base.GenericBenchmark(
        input_fn=unflatten_input_fn,
        op_name="unflatten",
        torch_op=torch.unflatten,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
