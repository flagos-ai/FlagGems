import os
import subprocess
import sys
import textwrap

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes covering 0-dim, 1-D, multi-D and large flatten spans.
PUT_SHAPES = [
    (),
    (1,),
    (16,),
    (64, 64),
    (20, 320, 15),
    (16, 128, 64, 60),
]

# `put` is dtype-agnostic in ATen: it accepts every integer, floating-point, bool
# and complex dtype. Cover all of them, not just the float set.
PUT_INT_DTYPES = utils.ALL_INT_DTYPES + [torch.int8, torch.uint8]
# `complex32` is excluded on purpose: ATen has no `put_cuda` for ComplexHalf
# (`NotImplementedError: "put_cuda" not implemented for 'ComplexHalf'`), so there
# is no reference to compare against.
PUT_COMPLEX_DTYPES = [torch.complex64, torch.complex128]
PUT_ALL_DTYPES = (
    utils.ALL_FLOAT_DTYPES + PUT_INT_DTYPES + utils.BOOL_TYPES + PUT_COMPLEX_DTYPES
)

# Accumulating many low-precision values via reordered atomic adds accumulates
# rounding error; use a looser tolerance than the default `1e-4` for these cases.
_ACCUMULATE_ATOL = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-1,
    torch.float32: 1e-4,
}


def accumulate_atol(dtype):
    return _ACCUMULATE_ATOL.get(dtype, 1e-4)


def is_exact(dtype):
    """Integer, bool and complex-integer results must match bit-for-bit."""
    return not dtype.is_floating_point and not dtype.is_complex


def gen_input(shape, dtype, device):
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(shape, dtype=dtype, device=device, requires_grad=False)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device).to(dtype)
    # Keep the magnitudes small so an accumulate over many duplicates stays inside
    # the narrow integer dtypes and the comparison is not about overflow wrap.
    return torch.randint(-8, 8, shape, device=device).to(dtype)


def gen_source(count, dtype, device):
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(count, dtype=dtype, device=device, requires_grad=False)
    if dtype == torch.bool:
        return torch.randint(0, 2, (count,), device=device).to(dtype)
    return torch.randint(-8, 8, (count,), device=device).to(dtype)


def gen_index_and_source(inp, dtype, device, accumulate, count=None):
    numel = inp.numel()
    if count is None:
        # Use a subset of positions to exercise partial writes.
        count = max(1, numel // 3)
    if accumulate:
        # Duplicate indices are only meaningful with `accumulate=True`, where
        # atomic-add makes the result order-independent. Draw with replacement so
        # duplicates really occur and the sum has to be right.
        index = torch.randint(0, numel, (count,), dtype=torch.int64, device=device)
    else:
        # With `accumulate=False` and duplicate indices ATen's result is
        # unspecified -- the last writer wins and which one that is depends on
        # thread scheduling (measured: not bit-stable across 200 identical runs).
        # Draw unique indices so the expected answer is well defined.
        index = torch.randperm(numel, device=device)[:count].to(torch.int64)
    source = gen_source(count, dtype, device)
    return index, source


def assert_put_close(res, ref, dtype, accumulate=False):
    if is_exact(dtype):
        utils.gems_assert_equal(res, ref)
    elif accumulate:
        utils.gems_assert_close(res, ref, dtype, atol=accumulate_atol(dtype))
    else:
        utils.gems_assert_close(res, ref, dtype)


# ---------------------------------------------------------------------------
# put (out-of-place)
# ---------------------------------------------------------------------------
@pytest.mark.put
@pytest.mark.parametrize("shape", PUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put(shape, dtype, accumulate):
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source, accumulate=accumulate)
    res_out = flag_gems.put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)


@pytest.mark.put
@pytest.mark.parametrize("shape", [(16,), (64, 64)])
@pytest.mark.parametrize("dtype", PUT_ALL_DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_dtypes(shape, dtype, accumulate):
    # ATen accepts integer, floating-point, bool and complex `put`; check parity
    # for all of them under both accumulate modes.
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source, accumulate=accumulate)
    res_out = flag_gems.put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)


@pytest.mark.put
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.int32, torch.int8, torch.uint8, torch.bool]
)
def test_put_accumulate_duplicates(dtype):
    # Every slot is written many times over: a plain store would silently keep one
    # contribution and drop the rest, so this pins the atomic-add path. The narrow
    # integer dtypes have no `tl.atomic_add` lowering and go through an int32
    # staging buffer; ATen wraps on overflow and so must we.
    numel = 17
    repeats = 512
    inp = gen_input((numel,), dtype, flag_gems.device)
    index = torch.arange(numel, device=flag_gems.device).repeat(repeats)
    source = torch.ones(index.numel(), dtype=dtype, device=flag_gems.device)

    ref_out = torch.put(
        utils.to_reference(inp),
        utils.to_reference(index),
        utils.to_reference(source),
        accumulate=True,
    )
    res_out = flag_gems.put(inp, index, source, accumulate=True)

    assert_put_close(res_out, ref_out, dtype, accumulate=True)


@pytest.mark.put
@pytest.mark.parametrize("shape", PUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put_negative_index(shape, dtype):
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)

    numel = inp.numel()
    count = max(1, numel // 2)
    # Unique indices so the non-accumulating overwrite is order-independent;
    # shift half of them to the negative range to exercise negative indexing.
    perm = torch.randperm(numel, device=flag_gems.device)[:count].to(torch.int64)
    neg_mask = torch.arange(count, device=flag_gems.device) % 2 == 0
    index = torch.where(neg_mask, perm - numel, perm)
    source = gen_source(count, dtype, flag_gems.device)

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source)
    res_out = flag_gems.put(inp, index, source)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.put
@pytest.mark.parametrize("numel", [1, 2, 3, 5, 6, 7, 8, 64])
@pytest.mark.parametrize("index_at", ["first", "last", "neg_first", "neg_last"])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_index_bounds(numel, index_at, accumulate):
    # The exact ends of the valid range: 0, numel-1, -numel and -1. Each is put in
    # its own call because -numel and 0 alias the same slot (as do -1 and
    # numel-1), which would otherwise make this a duplicate-index case.
    dtype = torch.float32
    value = {"first": 0, "last": numel - 1, "neg_first": -numel, "neg_last": -1}[
        index_at
    ]
    inp = gen_input((numel,), dtype, flag_gems.device)
    index = torch.tensor([value], dtype=torch.int64, device=flag_gems.device)
    source = gen_source(1, dtype, flag_gems.device)

    ref_out = torch.put(
        utils.to_reference(inp),
        utils.to_reference(index),
        utils.to_reference(source),
        accumulate=accumulate,
    )
    res_out = flag_gems.put(inp, index, source, accumulate=accumulate)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.put
@pytest.mark.parametrize("numel", [1, 8, 1000])
@pytest.mark.parametrize("bad_offset", [0, 5])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_index_out_of_range(numel, bad_offset, negative, accumulate):
    # ATen rejects any index outside [-numel, numel): the CUDA op trips
    # `cuda_take_put_kernel() index out of bounds` as a *device-side* assert.
    # The bounds failure must stay on the device -- on the success path there is
    # no host round trip -- so this runs in a subprocess (a tripped device
    # assertion poisons the CUDA context for the rest of the process) and
    # asserts both the failure and the absence of a host-side sync.
    bad = -numel - 1 - bad_offset if negative else numel + bad_offset
    code = textwrap.dedent(f"""
        import torch
        import flag_gems

        numel, bad, accumulate = {numel}, {bad}, {accumulate}
        inp = torch.randn(numel, dtype=torch.float32, device=flag_gems.device)
        index = torch.tensor([0, bad], dtype=torch.int64, device=flag_gems.device)
        source = torch.randn(2, dtype=torch.float32, device=flag_gems.device)

        # Success path must not synchronize (no host round trip on a valid call).
        flag_gems.put(inp, index[:1], source[:1], accumulate=accumulate)

        out = flag_gems.put(inp, index, source, accumulate=accumulate)
        torch.cuda.synchronize()
        print("NO_ERROR")
        """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert "NO_ERROR" not in proc.stdout, (
        "an out-of-range index was silently accepted; "
        f"stdout={proc.stdout!r} stderr={proc.stderr[-300:]!r}"
    )
    assert "device-side assert" in proc.stderr or "trap" in proc.stderr.lower(), (
        "expected a device-side failure, got: " + proc.stderr[-300:]
    )


@pytest.mark.put
@pytest.mark.parametrize("shape", [(64, 64), (20, 320, 15), (16, 128, 64, 60)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put_non_contiguous(shape, dtype):
    # A transposed (non-contiguous) tensor exercises the multi-dim offset path.
    inp = gen_input(shape, dtype, flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate=False)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source)
    res_out = flag_gems.put(inp, index, source)

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out.stride() == ref_out.stride()


@pytest.mark.put
@pytest.mark.parametrize(
    "shape, perm",
    [
        ((2, 3, 2, 3, 2), (4, 0, 3, 1, 2)),
        ((2, 3, 2, 3, 2, 3), (5, 0, 2, 1, 4, 3)),
        ((2, 2, 2, 3, 2, 2, 2), (6, 0, 5, 1, 4, 2, 3)),
    ],
)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_non_contiguous_high_rank(shape, perm, accumulate):
    # `put` places no limit on rank, so a permuted 6-D or 7-D `self` must work
    # rather than trip a rank assertion.
    dtype = torch.float32
    inp = gen_input(shape, dtype, flag_gems.device).permute(perm)
    ref_inp = utils.to_reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source, accumulate=accumulate)
    res_out = flag_gems.put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)
    assert res_out.stride() == ref_out.stride()


@pytest.mark.put
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_storage_offset(dtype, accumulate):
    # A view into the middle of a larger buffer. A kernel that assumed a zero
    # storage offset would corrupt the elements on either side of the view, so the
    # untouched remainder of the base tensor is checked too.
    base = gen_input((64,), dtype, flag_gems.device)
    base_snapshot = base.clone()
    inp = base[7:31].reshape(4, 6).t()
    ref_inp = utils.to_reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source, accumulate=accumulate)
    res_out = flag_gems.put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)
    # `put` is out-of-place: neither the view nor its neighbours may change.
    assert torch.equal(base, base_snapshot)


@pytest.mark.put
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_non_contiguous_index_and_source(dtype, accumulate):
    # `index` and `source` are consumed in row-major flatten order, so a
    # transposed index/source pair must be read through its strides.
    inp = gen_input((64,), dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)

    if accumulate:
        index = (
            torch.randint(0, 64, (36,), dtype=torch.int64, device=flag_gems.device)
            .reshape(6, 6)
            .t()
        )
    else:
        index = (
            torch.randperm(64, device=flag_gems.device)[:36]
            .to(torch.int64)
            .reshape(6, 6)
            .t()
        )
    source = gen_source(36, dtype, flag_gems.device).reshape(6, 6).t()

    ref_out = torch.put(
        ref_inp,
        utils.to_reference(index),
        utils.to_reference(source),
        accumulate=accumulate,
    )
    res_out = flag_gems.put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)


@pytest.mark.put
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.complex64])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_empty_index(dtype, accumulate):
    # An empty index/source pair leaves `self` untouched.
    inp = gen_input((16,), dtype, flag_gems.device)
    index = torch.tensor([], dtype=torch.int64, device=flag_gems.device)
    source = torch.tensor([], dtype=dtype, device=flag_gems.device)

    ref_out = torch.put(
        utils.to_reference(inp),
        utils.to_reference(index),
        utils.to_reference(source),
        accumulate=accumulate,
    )
    res_out = flag_gems.put(inp, index, source, accumulate=accumulate)

    assert_put_close(res_out, ref_out, dtype, accumulate)


@pytest.mark.put
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_put_zero_sized_self(dtype):
    # An empty index into an empty `self` is a no-op; a non-empty one is an error.
    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    empty_index = torch.tensor([], dtype=torch.int64, device=flag_gems.device)
    empty_source = torch.tensor([], dtype=dtype, device=flag_gems.device)

    ref_out = torch.put(
        utils.to_reference(inp),
        utils.to_reference(empty_index),
        utils.to_reference(empty_source),
    )
    res_out = flag_gems.put(inp, empty_index, empty_source)
    utils.gems_assert_equal(res_out, ref_out)

    with pytest.raises(IndexError):
        flag_gems.put(
            inp,
            torch.tensor([0], dtype=torch.int64, device=flag_gems.device),
            gen_source(1, dtype, flag_gems.device),
        )


@pytest.mark.put
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.complex64])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_does_not_mutate_self(dtype, accumulate):
    # `put` is the out-of-place variant: `self` must come back untouched.
    for inp in (
        gen_input((64,), dtype, flag_gems.device),
        gen_input((8, 12), dtype, flag_gems.device).t(),
    ):
        snapshot = inp.clone()
        index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
        flag_gems.put(inp, index, source, accumulate=accumulate)
        assert torch.equal(inp, snapshot)


@pytest.mark.put
def test_put_device_mismatch():
    # ATen requires self/index/source to share a device and refuses to move them.
    inp = gen_input((16,), torch.float32, flag_gems.device)
    index = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)
    source = gen_source(1, torch.float32, flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.put(inp, index.cpu(), source)
    with pytest.raises(RuntimeError):
        flag_gems.put(inp, index, source.cpu())


@pytest.mark.put
def test_put_invalid_dtypes():
    inp = gen_input((16,), torch.float32, flag_gems.device)
    index = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)

    # index must be int64
    with pytest.raises(RuntimeError):
        flag_gems.put(
            inp,
            index.to(torch.int32),
            gen_source(1, torch.float32, flag_gems.device),
        )
    # source must share self's dtype
    with pytest.raises(RuntimeError):
        flag_gems.put(inp, index, gen_source(1, torch.float64, flag_gems.device))
    # index and source must have the same number of elements
    with pytest.raises(IndexError):
        flag_gems.put(
            inp,
            torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device),
            gen_source(1, torch.float32, flag_gems.device),
        )


@pytest.mark.put
@pytest.mark.parametrize("shape", [(64,), (64, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put_index_source_diff_shapes(shape, dtype):
    # index and source may have different shapes as long as they share numel.
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)

    count = max(1, inp.numel() // 2)
    # Unique indices for a deterministic, order-independent overwrite.
    index = torch.randperm(inp.numel(), device=flag_gems.device)[:count].to(torch.int64)
    # Reshape source to a different shape with the same number of elements.
    source = gen_source(count, dtype, flag_gems.device).reshape(
        (count // 4, 4) if count % 4 == 0 else (count,)
    )

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source)
    res_out = flag_gems.put(inp, index, source)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# put.out
# ---------------------------------------------------------------------------
@pytest.mark.put_out
@pytest.mark.parametrize("shape", PUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_out(shape, dtype, accumulate):
    # `torch.put` has no `out=` overload, so the reference is `torch.ops.aten.put.out`,
    # which is available on CPU as well as CUDA and so works under --ref=cpu too.
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    out = torch.empty_like(inp)
    ref_out = torch.empty_like(ref_inp)
    ref_out = torch.ops.aten.put.out(
        ref_inp, ref_index, ref_source, accumulate, out=ref_out
    )
    res_out = flag_gems.put_out(inp, index, source, accumulate, out=out)

    assert_put_close(res_out, ref_out, dtype, accumulate)
    assert_put_close(out, ref_out, dtype, accumulate)


@pytest.mark.put_out
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_out_non_contiguous_out(dtype, accumulate):
    # ATen writes into `out` through its own strides and leaves them alone.
    inp = gen_input((12, 8), dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    out = torch.empty((8, 12), dtype=dtype, device=flag_gems.device).t()
    ref_out = torch.empty_like(utils.to_reference(out))
    ref_out = torch.ops.aten.put.out(
        ref_inp, ref_index, ref_source, accumulate, out=ref_out
    )
    res_out = flag_gems.put_out(inp, index, source, accumulate, out=out)

    assert_put_close(res_out, ref_out, dtype, accumulate)
    assert res_out is out


@pytest.mark.put_out
def test_put_out_invalid():
    inp = gen_input((16,), torch.float32, flag_gems.device)
    index = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)
    source = gen_source(1, torch.float32, flag_gems.device)

    # `out` must share self's dtype and device.
    with pytest.raises(RuntimeError):
        flag_gems.put_out(
            inp,
            index,
            source,
            False,
            out=torch.empty(16, dtype=torch.float64, device=flag_gems.device),
        )
    with pytest.raises(RuntimeError):
        flag_gems.put_out(inp, index, source, False, out=torch.empty(16))


@pytest.mark.put
def test_put_complex32_rejected():
    # Native CUDA `put` has no ComplexHalf kernel. `complex32` used to be routed
    # through the FP16 real-view path, silently producing a result where ATen
    # raises; the two must agree on the exception type and message.
    inp = torch.zeros(4, dtype=torch.complex32, device=flag_gems.device)
    index = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)
    source = torch.ones(1, dtype=torch.complex32, device=flag_gems.device)

    # Compare against the native call on the same device so the message and the
    # exception type are checked against the CUDA kernel, not the CPU fallback.
    with pytest.raises(NotImplementedError) as ref_err:
        torch.put(inp.clone(), index, source)
    with pytest.raises(NotImplementedError) as res_err:
        flag_gems.put(inp, index, source)

    assert str(ref_err.value) == str(res_err.value)
