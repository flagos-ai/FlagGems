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
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

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

    for i, shape in enumerate(indices_shape):
        # np.random.choice can accept tuple as size parameter
        index = np.random.choice(
            np.arange(input_shape[i]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, device=flag_gems.device))

    return indices


@pytest.mark.index_put
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_FALSE
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_put_acc_false(input_shape, indices_shape, values_shape, is_bool, dtype):
    accumulate = False
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
        if flag_gems.vendor_name == "tsingmicro":
            K = indices[0].to(device="cpu").sum().item()
        else:
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
    ref_out = torch.index_put(ref_inp, ref_indices, ref_values, accumulate)
    out = flag_gems.index_put(inp, indices, values, accumulate)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index_put
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_TRUE
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_put_acc_true(input_shape, indices_shape, values_shape, is_bool, dtype):
    utils.init_seed(0)

    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(24)
        torch.mlu.manual_seed_all(24)

    accumulate = True
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
        if flag_gems.vendor_name == "tsingmicro":
            K = indices[0].to(device="cpu").sum().item()
        else:
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
    ref_out = torch.index_put(ref_inp, ref_indices, ref_values, accumulate)
    out = flag_gems.index_put(inp, indices, values, accumulate)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.index_put_
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_FALSE
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_put__acc_false(input_shape, indices_shape, values_shape, is_bool, dtype):
    accumulate = False
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
        if flag_gems.vendor_name == "tsingmicro":
            K = indices[0].to(device="cpu").sum().item()
        else:
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
    torch.index_put_(ref_inp, ref_indices, ref_values, accumulate)
    flag_gems.index_put_(inp, indices, values, accumulate)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_put_
@pytest.mark.parametrize(
    "input_shape, indices_shape, values_shape, is_bool", INDEX_PUT_SHAPE_ACC_TRUE
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_put__acc_true(input_shape, indices_shape, values_shape, is_bool, dtype):
    utils.init_seed(0)

    if flag_gems.vendor_name == "metax":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    if flag_gems.vendor_name == "mthreads":
        torch.manual_seed(0)
        torch.musa.manual_seed_all(0)

    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    accumulate = True
    inp = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    indices = gen_indices_for_index_put(input_shape, indices_shape, accumulate, is_bool)

    if is_bool:
        if flag_gems.vendor_name == "tsingmicro":
            K = indices[0].to(device="cpu").sum().item()
        else:
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
    torch.index_put_(ref_inp, ref_indices, ref_values, accumulate)
    flag_gems.index_put_(inp, indices, values, accumulate)

    if flag_gems.vendor_name == "cambricon" and dtype == torch.float16:
        inp = utils.to_cpu(inp, ref_inp)
        ref_inp = ref_inp.to(dtype)
        torch.testing.assert_close(inp, ref_inp, atol=3e-3, rtol=3e-2)
    else:
        utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.index_put
@pytest.mark.parametrize("dtype", [torch.float32])
def test_index_put_error_all_none(dtype):
    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = [None, None]
    values = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)

    with pytest.raises(
        ValueError, match="At least one non-None index tensor is required"
    ):
        flag_gems.index_put(inp, indices, values, accumulate=False)


@pytest.mark.index_put_
@pytest.mark.parametrize("dtype", [torch.float32])
def test_index_put__error_all_none(dtype):
    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = [None, None]
    values = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)

    with pytest.raises(
        ValueError, match="At least one non-None index tensor is required"
    ):
        flag_gems.index_put_(inp, indices, values, accumulate=False)


# Format: (input_shape, indices_config)
# 0 in indices_config means a Tensor, 1 in indices_config means None
MIXED_INDEX_SHAPES = [
    ((1024, 1024), (0, 1)),
    ((1024, 1024), (1, 0)),
    ((32, 32, 32), (0, 0, 1)),
    ((32, 32, 32), (0, 1, 0)),
    ((32, 32, 32), (1, 0, 0)),
    ((64, 64, 64), (1, 0, 1)),
    ((12, 12, 12, 12), (1, 0, 0, 0)),
    ((12, 12, 12, 12), (0, 1, 0, 0)),
    ((16, 16, 16, 16), (1, 0, 0, 1)),
    ((16, 16, 16, 16), (0, 1, 1, 0)),
    ((8, 8, 8, 8), (0, 1, 1, 1)),
    ((8, 8, 8, 8), (1, 1, 0, 1)),
]


@pytest.mark.index_put
@pytest.mark.parametrize("input_shape, indices_config", MIXED_INDEX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_index_put_mixed_none_and_tensor(input_shape, indices_config, dtype):
    accumulate = False
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)

    tensor_dims = [
        input_shape[i] for i, is_none in enumerate(indices_config) if is_none == 0
    ]
    min_dim = min(tensor_dims)
    idx_len = random.randint(3, min(min_dim, 32))
    unique_pool = torch.randperm(min_dim, device=flag_gems.device)[:idx_len]

    indices, ref_indices = [], []
    for i, is_none in enumerate(indices_config):
        if is_none:
            indices.append(None)
            ref_indices.append(slice(None))
        else:
            indices.append(unique_pool)
            ref_indices.append(unique_pool.cpu())

    ref_inp = utils.to_reference(inp)
    target_shape = ref_inp[tuple(ref_indices)].shape

    values = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)
    ref_values = utils.to_reference(values)

    ref_out = ref_inp.clone()
    ref_out[tuple(ref_indices)] = ref_values

    out = flag_gems.index_put(inp, indices, values, accumulate)
    utils.gems_assert_close(out, ref_out, dtype)
