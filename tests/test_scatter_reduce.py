import pytest
import torch

import flag_gems
from tests.accuracy_utils import gems_assert_close, gems_assert_equal, to_reference

# ---------------------------------------------------------------------------
# Parametrize helpers
# ---------------------------------------------------------------------------

REDUCE_OPS = ["sum", "prod", "mean", "amax", "amin"]

SCATTER_REDUCE_SHAPES = [
    (16,),
    (128, 64),
    (32, 128, 64),
    (8, 32, 64, 32),
]

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
INT_SCATTER_DTYPES = [torch.int32, torch.int64]


def _make_scatter_reduce_inputs(shape, dim, dtype, device="cuda"):
    """Create compatible (self, index, src) tensors for scatter_reduce."""
    if dtype.is_floating_point:
        self_t = torch.randn(shape, dtype=dtype, device=device)
        src = torch.randn(shape, dtype=dtype, device=device)
    else:
        self_t = torch.randint(-10, 10, shape, dtype=dtype, device=device)
        src = torch.randint(-10, 10, shape, dtype=dtype, device=device)

    out_size_dim = shape[dim]
    index = torch.randint(0, out_size_dim, list(shape), device=device)
    return self_t, index, src


# ---------------------------------------------------------------------------
# Tests for scatter_reduce.two  (out-of-place)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SCATTER_REDUCE_SHAPES)
@pytest.mark.parametrize("reduce", REDUCE_OPS)
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_scatter_reduce_float(shape, reduce, include_self, dtype):
    dim = len(shape) - 1  # always reduce along last dim
    self_t, index, src = _make_scatter_reduce_inputs(shape, dim, dtype)

    ref_self = to_reference(self_t, upcast=True)
    ref_src = to_reference(src, upcast=True)
    ref_index = to_reference(index)

    ref_out = torch.scatter_reduce(
        ref_self, dim, ref_index, ref_src, reduce, include_self=include_self
    )

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            self_t, dim, index, src, reduce, include_self=include_self
        )

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.parametrize("shape", SCATTER_REDUCE_SHAPES)
@pytest.mark.parametrize("reduce", ["sum", "prod", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
@pytest.mark.parametrize("dtype", INT_SCATTER_DTYPES)
def test_scatter_reduce_int(shape, reduce, include_self, dtype):
    dim = len(shape) - 1
    self_t, index, src = _make_scatter_reduce_inputs(shape, dim, dtype)

    ref_self = to_reference(self_t.float())
    ref_src = to_reference(src.float())
    ref_index = to_reference(index)
    ref_out = torch.scatter_reduce(
        ref_self, dim, ref_index, ref_src, reduce, include_self=include_self
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            self_t, dim, index, src, reduce, include_self=include_self
        )

    gems_assert_equal(res_out, ref_out)


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("reduce", REDUCE_OPS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_reduce_dim(dim, reduce, dtype):
    shape = (16, 32, 16)
    self_t, index, src = _make_scatter_reduce_inputs(shape, dim % len(shape), dtype)

    ref_self = to_reference(self_t, upcast=True)
    ref_src = to_reference(src, upcast=True)
    ref_index = to_reference(index)

    ref_out = torch.scatter_reduce(
        ref_self, dim, ref_index, ref_src, reduce, include_self=True
    )

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            self_t, dim, index, src, reduce, include_self=True
        )

    gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Tests for scatter_reduce.two_out  (out-of-place with output tensor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SCATTER_REDUCE_SHAPES)
@pytest.mark.parametrize("reduce", REDUCE_OPS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_reduce_out(shape, reduce, dtype):
    dim = len(shape) - 1
    self_t, index, src = _make_scatter_reduce_inputs(shape, dim, dtype)
    out = torch.empty_like(self_t)

    ref_self = to_reference(self_t, upcast=True)
    ref_src = to_reference(src, upcast=True)
    ref_index = to_reference(index)
    ref_out = torch.empty_like(ref_self)
    torch.scatter_reduce(
        ref_self, dim, ref_index, ref_src, reduce, include_self=True, out=ref_out
    )

    with flag_gems.use_gems():
        torch.scatter_reduce(
            self_t, dim, index, src, reduce, include_self=True, out=out
        )

    gems_assert_close(out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_scatter_reduce_1d_sum():
    self_t = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    index = torch.tensor([0, 1, 0, 2], device="cuda")
    src = torch.tensor([10.0, 20.0, 30.0, 40.0], device="cuda")

    ref = torch.scatter_reduce(
        to_reference(self_t),
        0,
        to_reference(index),
        to_reference(src),
        "sum",
        include_self=True,
    )

    with flag_gems.use_gems():
        res = torch.scatter_reduce(self_t, 0, index, src, "sum", include_self=True)

    gems_assert_close(res, ref, torch.float32)


def test_scatter_reduce_1d_include_self_false():
    self_t = torch.tensor([100.0, 200.0, 300.0], device="cuda")
    index = torch.tensor([0, 1, 0], device="cuda")
    src = torch.tensor([1.0, 2.0, 3.0], device="cuda")

    ref = torch.scatter_reduce(
        to_reference(self_t),
        0,
        to_reference(index),
        to_reference(src),
        "sum",
        include_self=False,
    )

    with flag_gems.use_gems():
        res = torch.scatter_reduce(self_t, 0, index, src, "sum", include_self=False)

    gems_assert_close(res, ref, torch.float32)


def test_scatter_reduce_mean_include_self():
    self_t = torch.ones(4, device="cuda")
    index = torch.tensor([0, 0, 1, 2], device="cuda")
    src = torch.tensor([3.0, 5.0, 7.0, 9.0], device="cuda")

    ref = torch.scatter_reduce(
        to_reference(self_t),
        0,
        to_reference(index),
        to_reference(src),
        "mean",
        include_self=True,
    )

    with flag_gems.use_gems():
        res = torch.scatter_reduce(self_t, 0, index, src, "mean", include_self=True)

    gems_assert_close(res, ref, torch.float32)


@pytest.mark.parametrize("reduce", REDUCE_OPS)
def test_scatter_reduce_1x1(reduce):
    shape = (1, 1)
    self_t, index, src = _make_scatter_reduce_inputs(shape, 0, torch.float32)
    ref = torch.scatter_reduce(
        to_reference(self_t),
        0,
        to_reference(index),
        to_reference(src),
        reduce,
        include_self=True,
    )
    with flag_gems.use_gems():
        res = torch.scatter_reduce(self_t, 0, index, src, reduce, include_self=True)
    gems_assert_close(res, ref, torch.float32)


@pytest.mark.parametrize("reduce", REDUCE_OPS)
def test_scatter_reduce_large_4096(reduce):
    shape = (4096, 4096)
    self_t = torch.randn(shape, dtype=torch.float32, device="cuda")
    src = torch.randn(shape, dtype=torch.float32, device="cuda")
    index = torch.randint(0, shape[-1], shape, dtype=torch.long, device="cuda")
    ref = torch.scatter_reduce(
        to_reference(self_t),
        -1,
        to_reference(index),
        to_reference(src),
        reduce,
        include_self=True,
    )
    with flag_gems.use_gems():
        res = torch.scatter_reduce(self_t, -1, index, src, reduce, include_self=True)
    gems_assert_close(res, ref, torch.float32)


def test_scatter_reduce_empty_index():
    self_t = torch.randn(8, dtype=torch.float32, device="cuda")
    src = torch.empty(0, dtype=torch.float32, device="cuda")
    index = torch.empty(0, dtype=torch.long, device="cuda")
    ref = torch.scatter_reduce(
        to_reference(self_t),
        0,
        to_reference(index),
        to_reference(src),
        "sum",
        include_self=True,
    )
    with flag_gems.use_gems():
        res = torch.scatter_reduce(self_t, 0, index, src, "sum", include_self=True)
    gems_assert_close(res, ref, torch.float32)


@pytest.mark.parametrize("reduce", ["sum", "amax", "amin"])
def test_scatter_reduce_with_inf(reduce):
    self_t = torch.tensor([1.0, float("inf"), 3.0, float("-inf")], device="cuda")
    src = torch.tensor([10.0, 20.0, 30.0, float("inf")], device="cuda")
    index = torch.tensor([0, 1, 2, 3], device="cuda")
    ref = torch.scatter_reduce(
        to_reference(self_t),
        0,
        to_reference(index),
        to_reference(src),
        reduce,
        include_self=True,
    )
    with flag_gems.use_gems():
        res = torch.scatter_reduce(self_t, 0, index, src, reduce, include_self=True)
    res_cpu = res.to("cpu") if ref.device == torch.device("cpu") else res
    torch.testing.assert_close(res_cpu, ref, equal_nan=True)


def test_scatter_reduce_with_nan():
    self_t = torch.tensor([1.0, float("nan"), 3.0], device="cuda")
    src = torch.tensor([10.0, 20.0, 30.0], device="cuda")
    index = torch.tensor([0, 1, 2], device="cuda")
    ref = torch.scatter_reduce(
        to_reference(self_t),
        0,
        to_reference(index),
        to_reference(src),
        "sum",
        include_self=True,
    )
    with flag_gems.use_gems():
        res = torch.scatter_reduce(self_t, 0, index, src, "sum", include_self=True)
    res_cpu = res.to("cpu") if ref.device == torch.device("cpu") else res
    assert torch.isnan(res_cpu[1]) and torch.isnan(ref[1])
    other = torch.tensor([i for i in range(3) if i != 1], device=ref.device)
    gems_assert_close(res_cpu[other], ref[other], torch.float32)


def test_scatter_reduce_invalid_reduce_raises():
    self_t = torch.zeros(4, dtype=torch.float32, device="cuda")
    src = torch.zeros(4, dtype=torch.float32, device="cuda")
    index = torch.zeros(4, dtype=torch.long, device="cuda")
    with flag_gems.use_gems():
        with pytest.raises(AssertionError, match="invalid reduce"):
            torch.scatter_reduce(self_t, 0, index, src, "not-a-real-reduce")
