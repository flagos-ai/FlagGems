import pytest
import torch

from . import base, consts


def permute_copy_input_fn(shape, dtype, device):
    x = base.generate_tensor_input(shape, dtype, device)
    # Generate appropriate dims based on rank
    ndim = len(shape)
    if ndim == 1:
        dims = (0,)
    elif ndim == 2:
        dims = (1, 0)
    elif ndim >= 3:
        # Create a non-trivial permutation: move last dim to front, 0 to back
        dims = tuple([ndim - 1] + list(range(1, ndim - 1)) + [0])
    yield x, dims


@pytest.mark.permute_copy
def test_permute_copy():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="permute_copy",
        torch_op=torch.permute_copy,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=permute_copy_input_fn,
    )
    bench.run()
