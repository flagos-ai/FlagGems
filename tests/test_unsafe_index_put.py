import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "sunrise", reason="Issues #3836: To Fix (Runtime Or LLVM)"
)


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    INT_DTYPES = [torch.int32, torch.int64, torch.int16, torch.int8, torch.uint8]

ALL_INPUT_DTYPES = FLOAT_DTYPES + INT_DTYPES + [torch.bool]
INDEX_DTYPES = [torch.int32, torch.int64]

# Test shapes: (input_shape, [indices_shapes...], values_shape, accumulate)
UNSAFE_INDEX_PUT_SHAPES = (
    ((32,), ((8,),), (8,), False),
    ((100,), ((100,),), (100,), True),
    ((32, 32), ((8,), (8,)), (8,), False),
    ((32, 32), ((8,), (2, 8)), (8,), False),
    ((32, 32), ((2, 8),), (32,), False),
    ((64, 64, 64), ((2, 8), (2, 8), (2, 8)), (2, 8), False),
    ((100,), ((100,),), (100,), True),
    ((32, 32), ((32, 32),), (32, 32, 32), True),
    ((16, 16, 4), ((16,),), (16, 16, 4), False),
)


def _make_input(shape, dtype, device, is_float):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device).to(torch.bool)
    elif is_float:
        return torch.randn(shape, dtype=dtype, device=device)
    else:
        return torch.randint(1, 10, shape, device=device).to(dtype)


def _make_values(shape, dtype, device, is_float):
    return _make_input(shape, dtype, device, is_float)


def gen_indices(input_shape, indices_shapes, accumulate, device, dtype=torch.int64):
    indices = []
    for i, shape in enumerate(indices_shapes):
        index = np.random.choice(
            np.arange(input_shape[i]), size=shape, replace=accumulate
        )
        indices.append(torch.tensor(index, dtype=dtype, device=device))
    return indices


def assert_accumulate_close(res, ref, dtype):
    """Assert helper for accumulate=True float tests.

    With duplicate indices the summation order is implementation-defined:
    PyTorch's CUDA kernel accumulates serially after a radix sort, while
    FlagGems accumulates via fp32 atomic_add in arbitrary GPU scheduling
    order. Both accumulate in fp32 and round to fp16/bf16 once, but fp32
    reassociation noise can flip the final rounding by 1 ulp of the output
    dtype. This is not an implementation defect — even PyTorch's own CPU vs
    CUDA results bit-differ on essentially every run for this case — so the
    tolerance must cover >=2 ulp. bf16's default rtol (0.016) already does;
    fp16's (0.001) does not.
    """
    if dtype == torch.float16:
        torch.testing.assert_close(res, ref, atol=1e-3, rtol=4e-3)
    else:
        utils.gems_assert_close(res, ref, dtype)


# ---- Core dtype × index_dtype tests ----
@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("inp_dtype", ALL_INPUT_DTYPES)
@pytest.mark.parametrize("idx_dtype", INDEX_DTYPES)
def test_unsafe_index_put_dtypes(inp_dtype, idx_dtype):
    """Test all input and index dtype combinations."""
    shape = (32, 32)
    is_float = inp_dtype.is_floating_point

    inp = _make_input(shape, inp_dtype, flag_gems.device, is_float)
    indices = gen_indices(shape, [(2, 8)], False, flag_gems.device, idx_dtype)
    values = _make_values((32,), inp_dtype, flag_gems.device, is_float)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx) for idx in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, False)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, False)

    if is_float:
        utils.gems_assert_close(res_out, ref_out, inp_dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


# ---- Bool mask index tests ----
@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("inp_dtype", ALL_INPUT_DTYPES)
def test_unsafe_index_put_bool_mask_dtypes(inp_dtype):
    """Test bool mask indices with all input dtypes."""
    shape = (16, 16)
    is_float = inp_dtype.is_floating_point

    inp = _make_input(shape, inp_dtype, flag_gems.device, is_float)
    mask = torch.randint(0, 2, (16,), dtype=torch.bool, device=flag_gems.device)
    K = mask.sum().item()
    if K == 0:
        pytest.skip("empty bool mask")
    values = _make_values((K, 16), inp_dtype, flag_gems.device, is_float)

    ref_inp = utils.to_reference(inp)
    ref_mask = utils.to_reference(mask)
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, [ref_mask], ref_values, False)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, [mask], values, False)

    if is_float:
        utils.gems_assert_close(res_out, ref_out, inp_dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


# ---- Shape parametrized tests (floats) ----
@pytest.mark.unsafe_index_put
@pytest.mark.parametrize(
    "input_shape, indices_shapes, values_shape, accumulate",
    UNSAFE_INDEX_PUT_SHAPES,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_unsafe_index_put_shapes(
    input_shape, indices_shapes, values_shape, accumulate, dtype
):
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    indices = gen_indices(input_shape, indices_shapes, accumulate, flag_gems.device)
    values = torch.randn(values_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx) for idx in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, accumulate)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, accumulate)

    if accumulate:
        assert_accumulate_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


# ---- Integer shape tests ----
@pytest.mark.unsafe_index_put
@pytest.mark.parametrize(
    "input_shape, indices_shapes, values_shape, accumulate",
    UNSAFE_INDEX_PUT_SHAPES[:5],
)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_unsafe_index_put_int_shapes(
    input_shape, indices_shapes, values_shape, accumulate, dtype
):
    inp = torch.randint(1, 10, input_shape, device=flag_gems.device).to(dtype)
    indices = gen_indices(input_shape, indices_shapes, accumulate, flag_gems.device)
    values = torch.randint(1, 10, values_shape, device=flag_gems.device).to(dtype)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx) for idx in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, accumulate)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, accumulate)

    utils.gems_assert_equal(res_out, ref_out)


# ---- Specific behavior tests ----
@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_unsafe_index_put_negative_indices(dtype):
    inp = torch.randn((32, 64), dtype=dtype, device=flag_gems.device)
    indices = [torch.tensor([-1, -5, -10], device=flag_gems.device)]
    values = torch.randn((3, 64), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx) for idx in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, False)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.unsafe_index_put
def test_unsafe_index_put_functional():
    inp = torch.randn((32, 32), device=flag_gems.device)
    inp_copy = inp.clone()
    indices = [torch.randint(0, 32, (8,), device=flag_gems.device)]
    values = torch.randn((8, 32), device=flag_gems.device)

    with flag_gems.use_gems():
        out = torch._unsafe_index_put(inp, indices, values, False)

    assert torch.equal(inp, inp_copy), "Input tensor was modified"
    assert not torch.equal(out, inp), "Output should differ from input"


@pytest.mark.unsafe_index_put
def test_unsafe_index_put_scalar_value():
    inp = torch.randn((16, 16), device=flag_gems.device)
    indices = [torch.tensor([0, 5, 10], device=flag_gems.device)]
    values = torch.tensor(3.14, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx) for idx in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, False)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, False)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_unsafe_index_put_accumulate(dtype):
    inp = torch.randn((32, 32), dtype=dtype, device=flag_gems.device)
    indices = [torch.randint(0, 32, (64,), device=flag_gems.device)]
    values = torch.randn((64, 32), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx) for idx in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, True)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, True)

    assert_accumulate_close(res_out, ref_out, dtype)


@pytest.mark.unsafe_index_put
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_unsafe_index_put_int_accumulate(dtype):
    inp = torch.randint(1, 5, (32, 32), device=flag_gems.device).to(dtype)
    indices = [torch.randint(0, 32, (64,), device=flag_gems.device)]
    values = torch.randint(1, 5, (64, 32), device=flag_gems.device).to(dtype)

    ref_inp = utils.to_reference(inp)
    ref_indices = [utils.to_reference(idx) for idx in indices]
    ref_values = utils.to_reference(values)
    ref_out = torch._unsafe_index_put(ref_inp, ref_indices, ref_values, True)

    with flag_gems.use_gems():
        res_out = torch._unsafe_index_put(inp, indices, values, True)

    utils.gems_assert_equal(res_out, ref_out)
