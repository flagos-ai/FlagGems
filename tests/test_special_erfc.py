import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_erfc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_erfc(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_erfc(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.special_erfc(ref_x)
    act_out = flag_gems.special_erfc(x)
    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.erfc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfc(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.erfc(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.erfc(ref_x)
    act_out = flag_gems.erfc(x)
    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.erfc_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfc_(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        # Compute the reference in fp32 and cast back, mirroring the
        # non-inplace erfc tests. The inplace kernel writes through
        # out0=x and is bitwise-identical to the non-inplace one, so
        # low-precision dtypes are covered here too (issue #4076).
        ref_out = torch.ops.aten.erfc_(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.erfc_(ref_x)
    act_out = flag_gems.erfc_(x)
    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(x, ref_out, dtype, equal_nan=True)


@pytest.mark.erfc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_erfc_byte(shape, dtype):
    # torch promotes integral erfc inputs to float32.
    x = torch.randint(1, 8, shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    ref_out = torch.ops.aten.erfc(ref_x)
    act_out = flag_gems.erfc(x)
    utils.gems_assert_close(act_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.erfc_
def test_erfc_int_rejects_inplace():
    # torch.erfc_ rejects integral inputs -- "result type Float can't be
    # cast to the desired output type Int"; so does FlagGems.
    x = torch.tensor([1, 2, 3], dtype=torch.int32, device=flag_gems.device)
    ref_x = x.cpu()
    with pytest.raises(RuntimeError):
        torch.ops.aten.erfc_(ref_x)
    with pytest.raises(RuntimeError):
        flag_gems.erfc_(x)


# ---------------------------------------------------------------------------
# Extended coverage: value-range grid x shape x dtype, scalar params, broadcast.
# ---------------------------------------------------------------------------

# Five value ranges per the coverage spec, over [-1, 1] plus the dtype edges.
ERFC_VALUE_RANGES = [
    ("[-1,1]", -1.0, 1.0),
    ("[0,1]", 0.0, 1.0),
    ("[-1,0]", -1.0, 0.0),
    ("[0,max]", 0.0, None),
    ("[min,0]", None, 0.0),
]


def _make_erfc_input(shape, dtype, lo, hi):
    """Build a float tensor in [lo, hi] on the active device."""
    if lo is None or hi is None:
        info = torch.finfo(dtype)
        lo = info.min if lo is None else lo
        hi = info.max if hi is None else hi
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(lo, hi)
    return inp


@pytest.mark.erfc
@pytest.mark.parametrize("rng", ERFC_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfc_value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    x = _make_erfc_input(shape, dtype, lo, hi)
    ref_x = utils.to_reference(x)

    ref_out = torch.special.erfc(ref_x)
    res_out = flag_gems.erfc(x)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Scalar params sweep: bool, int boundaries, float boundaries incl. nan/inf.
ERFC_PARAMS = [
    True,
    False,
    0,
    1,
    -1,
    255,
    -255,
    2**31 - 1,
    -(2**31),
    2**63 - 1,
    -(2**63),
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.5,
    -2.5,
    float("inf"),
    float("-inf"),
    float("nan"),
]


@pytest.mark.erfc
@pytest.mark.parametrize("val", ERFC_PARAMS)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfc_params(shape, val, dtype):
    # int64-range scalars overflow at::Half on GPU init, matching ATen's
    # integer-scalar conversion window; skip those combos.
    if dtype is torch.float16 and isinstance(val, int) and not isinstance(val, bool):
        if not -(2**15) <= val <= 2**15 - 1:
            pytest.skip("int64-range scalar overflows fp16 in ATen")
    x = torch.full(shape, float(val), dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    ref_out = torch.special.erfc(ref_x)
    res_out = flag_gems.erfc(x)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Broadcast shapes: 1-D and multi-D broadcasts against the 4D base.
ERFC_BROADCAST_CASES = [
    ((16, 128, 64, 60), (60,)),
    ((16, 128, 64, 60), (1, 60)),
    ((16, 128, 64, 60), (1, 1, 64, 60)),
    ((16, 128, 64, 60), (16, 1, 1, 1)),
    ((20, 320, 15), (15,)),
    ((1024, 1024), (1,)),
]


@pytest.mark.erfc
@pytest.mark.parametrize("shapes", ERFC_BROADCAST_CASES, ids=lambda c: str(c))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfc_broadcast(shapes, dtype):
    # Unary op: broadcast means a stride-0 expanded view of the small tensor.
    shape, other_shape = shapes
    x = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    expanded = x.expand(shape)
    ref_x = utils.to_reference(expanded)

    ref_out = torch.special.erfc(ref_x)
    res_out = flag_gems.erfc(expanded)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
