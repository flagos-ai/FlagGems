# Tests for tril / tril_: 2D shapes, diagonals, 3D/4D batch, non-square,
# edge diagonals, in-place; dtypes float16/32/bfloat16/int16/32/64.

import pytest
import torch

import flag_gems

FLOAT_DTYPES = [torch.float16, torch.float32, torch.bfloat16]
INT_DTYPES = [torch.int16, torch.int32, torch.int64]
ALL_DTYPES = FLOAT_DTYPES + INT_DTYPES

ATOL = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
    torch.int16: 0,
    torch.int32: 0,
    torch.int64: 0,
}
RTOL = {
    torch.float16: 1e-4,
    torch.float32: 1e-4,
    torch.bfloat16: 1e-4,
    torch.int16: 0,
    torch.int32: 0,
    torch.int64: 0,
}

SHAPES_2D = [
    (1, 1),
    (8, 8),
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
]


def make_input(shape, dtype):
    if dtype in FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        iinfo = torch.iinfo(dtype)
        return torch.randint(
            iinfo.min // 2, iinfo.max // 2, shape, dtype=dtype, device=flag_gems.device
        )


def assert_close(result, reference, dtype):
    if dtype in INT_DTYPES:
        assert torch.equal(
            result, reference
        ), f"Integer tril mismatch for dtype={dtype}"
    else:
        torch.testing.assert_close(
            result.to(torch.float32),
            reference.to(torch.float32),
            atol=ATOL[dtype],
            rtol=RTOL[dtype],
        )


# ── tril (out-of-place) ────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", SHAPES_2D)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tril_2d_shapes(shape, dtype):
    """Standard 2D shapes across all dtypes, diagonal=0."""
    inp = make_input(shape, dtype)
    ref = torch.tril(inp, 0)
    with flag_gems.use_gems():
        res = torch.tril(inp, 0)
    assert_close(res, ref, dtype)


@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("shape", [(64, 64), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_tril_diagonals(shape, dtype, diagonal):
    """Parametrised diagonal offsets for both float and integer types."""
    inp = make_input(shape, dtype)
    ref = torch.tril(inp, diagonal)
    with flag_gems.use_gems():
        res = torch.tril(inp, diagonal)
    assert_close(res, ref, dtype)


@pytest.mark.parametrize("shape", [(4, 64, 64), (8, 32, 32)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.bfloat16, torch.int32]
)
def test_tril_3d_batch(shape, dtype):
    """3D batched tensors."""
    inp = make_input(shape, dtype)
    ref = torch.tril(inp, 0)
    with flag_gems.use_gems():
        res = torch.tril(inp, 0)
    assert_close(res, ref, dtype)


@pytest.mark.parametrize("shape", [(2, 3, 32, 32), (4, 4, 16, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_tril_4d_batch(shape, dtype):
    """4D batched tensors."""
    inp = make_input(shape, dtype)
    ref = torch.tril(inp, 0)
    with flag_gems.use_gems():
        res = torch.tril(inp, 0)
    assert_close(res, ref, dtype)


@pytest.mark.parametrize("shape", [(128, 64), (64, 256), (1, 1024), (1024, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_tril_nonsquare(shape, dtype):
    """Non-square 2D matrices."""
    inp = make_input(shape, dtype)
    ref = torch.tril(inp, 0)
    with flag_gems.use_gems():
        res = torch.tril(inp, 0)
    assert_close(res, ref, dtype)


@pytest.mark.parametrize("diagonal", [-100, -63, 63, 100])
def test_tril_edge_diagonals(diagonal):
    """Extreme diagonal offsets that span the entire matrix or leave it empty."""
    dtype = torch.float32
    shape = (64, 64)
    inp = make_input(shape, dtype)
    ref = torch.tril(inp, diagonal)
    with flag_gems.use_gems():
        res = torch.tril(inp, diagonal)
    assert_close(res, ref, dtype)


# ── tril_ (in-place) ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", [(1, 1), (8, 8), (64, 64), (256, 256), (1024, 1024)])
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tril_inplace(shape, dtype):
    """In-place tril_ across standard shapes and all dtypes."""
    inp = make_input(shape, dtype)
    ref_inp = inp.clone()

    ref_inp.tril_(0)
    with flag_gems.use_gems():
        inp.tril_(0)

    assert_close(inp, ref_inp, dtype)


@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_tril_inplace_diagonals(dtype, diagonal):
    """In-place tril_ with various diagonal offsets."""
    shape = (64, 64)
    inp = make_input(shape, dtype)
    ref_inp = inp.clone()

    ref_inp.tril_(diagonal)
    with flag_gems.use_gems():
        inp.tril_(diagonal)

    assert_close(inp, ref_inp, dtype)


@pytest.mark.parametrize("shape", [(4, 64, 64), (2, 3, 32, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_tril_inplace_batch(shape, dtype):
    """In-place tril_ on batched tensors."""
    inp = make_input(shape, dtype)
    ref_inp = inp.clone()

    ref_inp.tril_(0)
    with flag_gems.use_gems():
        inp.tril_(0)

    assert_close(inp, ref_inp, dtype)
