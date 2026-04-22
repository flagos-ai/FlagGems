import random
import time

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    ATTN_HEADS = [2]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    ATTN_HEADS = [2, 4, 8, 16, 32]

INDEX_PUT_SHAPE_ACC_FALSE = (
    ((2**28,), ((2**16,),), (2**16,), False),
    ((32, 32), ((8,), (8,)), (8,), False),
    ((32, 32), ((8,), (2, 8)), (8,), False),
    ((32, 32), ((2, 8),), (32,), False),
    ((512, 512, 512), ((128,), (128,), (128,)), (128,), False),
    ((512, 512, 512), ((2, 128), (128,), (128,)), (128,), False),
    ((512, 512, 512), ((2, 128),), (512,), False),
    (
        (64, 64, 64),
        (
            (2, 8),
            (2, 8),
        ),
        (2, 8, 64),
        False,
    ),
    ((100,), ((100,),), (100,), True),
    ((32, 32), ((32, 32),), (32, 32), True),
    ((16, 16, 4), ((16, 16, 4),), (16, 16, 4), True),
)

INDEX_PUT_SHAPE_ACC_TRUE = (
    ((2**28,), ((2**16,),), (2**16,), False),
    ((32, 32), ((8,), (8,)), (8,), False),
    ((512, 512, 512), ((128,), (128,), (128,)), (128,), False),
    ((64, 64, 64), ((2, 8), (2, 8), (2, 8)), (2, 8), False),
    ((32, 32), ((32, 32),), (32 * 32,), True),
)


# Make sure every thread has same seed.
random.seed(time.time() // 100)


def gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool):
    """
    Generate indices for torch.index_put.
    This function supports multi-dimensional integer index shapes (e.g., (2, 8))
    when is_bool is False, and generates a single boolean mask tensor when
    is_bool is True. This is unlike gen_indices which is designed for
    torch.ops.aten.index that requires broadcastable indices.
    """
    indices = []

    if is_bool:
        mask_shape = indices_shape[0]

        mask = torch.randint(
            0, 2, size=mask_shape, dtype=torch.bool, device=flag_gems.device
        )
        return [mask]

    else:
        for i, shape in enumerate(indices_shape):
            # np.random.choice can accept tuple as size parameter
            index = np.random.choice(
                np.arange(input_shape[i]), size=shape, replace=accumulate
            )
            indices.append(torch.tensor(index, device=flag_gems.device))
        return indices


# Tests for _index_put_impl_
@pytest.mark.index_put_impl
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_FALSE
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__index_put_impl__acc_false(
    input_shape, indices_shape, values_shape, is_bool, dtype
):
    accumulate = False
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
        K = indices[0].sum().item()
        values = torch.randn(
            (K,), dtype=dtype, device=flag_gems.device, requires_grad=False
        )
    else:
        values = torch.randn(
            values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        )

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(index) for index in indices]
    ref_values = utils.to_reference(values)
    torch._index_put_impl_(ref_inp, ref_indices, ref_values, accumulate, unsafe=False)
    with flag_gems.use_gems():
        torch._index_put_impl_(inp, indices, values, accumulate, unsafe=False)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_put_impl
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_TRUE
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test__index_put_impl__acc_true(
    input_shape, indices_shape, values_shape, is_bool, dtype
):
    utils.init_seed(0)

    accumulate = True
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
        K = indices[0].sum().item()
        values = torch.randn(
            (K,), dtype=dtype, device=flag_gems.device, requires_grad=False
        )
    else:
        values = torch.randn(
            values_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
        )

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_indices = [utils.to_reference(index) for index in indices]
    ref_values = utils.to_reference(values, upcast=True)
    torch._index_put_impl_(ref_inp, ref_indices, ref_values, accumulate, unsafe=False)
    with flag_gems.use_gems():
        torch._index_put_impl_(inp, indices, values, accumulate, unsafe=False)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_put_impl
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("unsafe", [True, False])
def test__index_put_impl__unsafe_param(dtype, unsafe):
    """Test _index_put_impl_ with both unsafe=True and unsafe=False"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = [torch.randint(0, 32, (8,), device=flag_gems.device)]
    values = torch.randn((8, 64), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(index) for index in indices]
    ref_values = utils.to_reference(values)
    torch._index_put_impl_(
        ref_inp, ref_indices, ref_values, accumulate=False, unsafe=unsafe
    )
    with flag_gems.use_gems():
        torch._index_put_impl_(inp, indices, values, accumulate=False, unsafe=unsafe)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_put_impl
@pytest.mark.parametrize("dtype", [torch.float32])
def test__index_put_impl__error_all_none(dtype):
    """Test error handling: all None indices for _index_put_impl_"""

    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = [None, None]
    values = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)

    # PyTorch validates indices before dispatch, so TypeError is raised
    with pytest.raises(TypeError):
        with flag_gems.use_gems():
            torch._index_put_impl_(inp, indices, values, accumulate=False, unsafe=False)
