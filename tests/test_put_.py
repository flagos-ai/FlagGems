import os
import subprocess
import sys
import textwrap

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes covering 1-D, multi-D and large flatten spans. Mirrors `test_put.py`.
PUT_SHAPES = [
    (),
    (1,),
    (16,),
    (64, 64),
    (20, 320, 15),
    (16, 128, 64, 60),
]

# Accumulating many low-precision values via reordered atomic adds accumulates
# rounding error; use a looser tolerance than the default `1e-4` for these cases.
_ACCUMULATE_ATOL = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-1,
    torch.float32: 1e-4,
}


def accumulate_atol(dtype):
    return _ACCUMULATE_ATOL.get(dtype, 1e-4)


def gen_input(shape, dtype, device):
    return torch.randn(shape, dtype=dtype, device=device, requires_grad=False)


def gen_index_and_source(inp, dtype, device, accumulate, count=None):
    import numpy as np

    numel = inp.numel()
    if count is None:
        # Use a subset of positions to exercise partial writes.
        count = max(1, numel // 3)
    # When not accumulating, repeated indices race (PyTorch keeps the first
    # write in order while our kernel is unordered), so only allow repeats when
    # accumulating, where atomic-add makes the result order-independent.
    replace = bool(accumulate)
    index = np.random.choice(np.arange(numel), size=count, replace=replace)
    index = torch.tensor(index, dtype=torch.int64, device=device)
    source = torch.randn(count, dtype=dtype, device=device, requires_grad=False)
    return index, source


# ---------------------------------------------------------------------------
# put_ (in-place)
# ---------------------------------------------------------------------------
@pytest.mark.put_
@pytest.mark.parametrize("shape", PUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put_(shape, dtype, accumulate):
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_inp.put_(ref_index, ref_source, accumulate=accumulate)
    res = flag_gems.put_(inp, index, source, accumulate=accumulate)

    # `put_` is in-place: the return value must be `self` itself, not a copy.
    assert res is inp

    if accumulate:
        utils.gems_assert_close(inp, ref_inp, dtype, atol=accumulate_atol(dtype))
    else:
        utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("shape", PUT_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put__negative_index(shape, dtype):
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    numel = inp.numel()
    count = max(1, numel // 2)
    # Unique indices so the non-accumulating overwrite is order-independent;
    # shift half of them to the negative range to exercise negative indexing.
    perm = torch.randperm(numel, device=flag_gems.device)[:count].to(torch.int64)
    neg_mask = torch.arange(count, device=flag_gems.device) % 2 == 0
    index = torch.where(neg_mask, perm - numel, perm)
    source = torch.randn(count, dtype=dtype, device=flag_gems.device)

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_inp.put_(ref_index, ref_source)
    flag_gems.put_(inp, index, source)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("shape", [(64, 64), (20, 320, 15), (16, 128, 64, 60)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put__non_contiguous(shape, dtype):
    # A transposed (non-contiguous) tensor exercises the multi-dim offset path.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device).transpose(0, 1)
    ref_inp = utils.to_reference(inp.clone())

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate=False)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_inp.put_(ref_index, ref_source)
    flag_gems.put_(inp, index, source)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int16, torch.bool])
@pytest.mark.parametrize(
    "layout",
    ["transpose", "slice", "narrow_perm", "row_slice"],
)
def test_put__non_contiguous_narrow_accumulate(dtype, layout):
    """Non-contiguous narrow int/bool under ``accumulate=True``.

    These dtypes accumulate through an int32 staging buffer because Triton has
    no atomic for them. ``put_`` indexes ``self`` by its row-major *logical*
    flattening, so both the widen (reading ``self``) and the narrow (writing it
    back) must decode that flat offset against the real strides -- staging a
    strided ``self`` as if it were contiguous corrupts exactly the indexed
    elements.
    """
    utils.init_seed(0)
    base = _gen_typed((16, 24), dtype, flag_gems.device)
    inp = {
        "transpose": lambda t: t.t(),
        "slice": lambda t: t[:, ::2],
        "narrow_perm": lambda t: t[3:11, 2:18].t(),
        "row_slice": lambda t: t[::2],
    }[layout](base)
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp.clone())

    numel = inp.numel()
    index = torch.randint(
        0, numel, (numel * 3,), dtype=torch.int64, device=flag_gems.device
    )
    source = _gen_typed((index.numel(),), dtype, flag_gems.device)

    ref_inp.put_(utils.to_reference(index), utils.to_reference(source), accumulate=True)
    res = flag_gems.put_(inp, index, source, accumulate=True)

    assert res is inp
    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.put_
@pytest.mark.parametrize("shape", [(64,), (64, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put__index_source_diff_shapes(shape, dtype):
    # index and source may have different shapes as long as they share numel.
    inp = gen_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    count = max(1, inp.numel() // 2)
    # Unique indices for a deterministic, order-independent overwrite.
    index = torch.randperm(inp.numel(), device=flag_gems.device)[:count].to(torch.int64)
    # Reshape source to a different shape with the same number of elements.
    source = torch.randn(
        (count // 4, 4) if count >= 4 else (count,),
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_inp.put_(ref_index, ref_source)
    flag_gems.put_(inp, index, source)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_put__integer(dtype):
    inp = torch.zeros(100, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    import numpy as np

    n = inp.numel()
    idx_np = np.random.choice(np.arange(n), size=(50,), replace=False)
    index = torch.tensor(idx_np, dtype=torch.int64, device=flag_gems.device)
    source = torch.randint(0, 1000, (50,), dtype=dtype, device=flag_gems.device)

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)
    ref_inp.put_(ref_index, ref_source)
    flag_gems.put_(inp, index, source)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.put_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put__accumulate_duplicates(dtype):
    # Exercise the accumulate path with heavy index duplication, where the
    # kernel accumulates in the tensor's own (low-precision) dtype.
    utils.init_seed(0)
    inp = torch.randn(64, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    import numpy as np

    n = inp.numel()
    idx_np = np.random.choice(np.arange(n), size=(n * 8,), replace=True)
    index = torch.tensor(idx_np, dtype=torch.int64, device=flag_gems.device)
    source = torch.randn(n * 8, dtype=dtype, device=flag_gems.device)

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)
    ref_inp.put_(ref_index, ref_source, accumulate=True)
    flag_gems.put_(inp, index, source, accumulate=True)

    utils.gems_assert_close(inp, ref_inp, dtype, atol=accumulate_atol(dtype))


# ---------------------------------------------------------------------------
# dtype coverage: integer, bool and complex, with and without accumulate.
# ---------------------------------------------------------------------------
# `accumulate=True` has no Triton atomic for int8/uint8/int16/bool, so those
# dtypes take a staged int32 path; the parametrisation below covers both sides.
_PUT_ALL_DTYPES = (
    utils.ALL_FLOAT_DTYPES
    + utils.ALL_INT_DTYPES
    + [torch.int8, torch.uint8]
    + utils.BOOL_TYPES
    + [torch.complex64, torch.complex128]
)


def _gen_typed(shape, dtype, device):
    """Random tensor of any dtype `put_` accepts."""
    if dtype is torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool, device=device)
    if dtype.is_complex:
        real = torch.randn(shape, device=device)
        imag = torch.randn(shape, device=device)
        return (real + 1j * imag).to(dtype)
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    info = torch.iinfo(dtype)
    lo = max(info.min, -100)
    hi = min(info.max, 100)
    return torch.randint(lo, hi, shape, dtype=dtype, device=device)


@pytest.mark.put_
@pytest.mark.parametrize("dtype", _PUT_ALL_DTYPES)
@pytest.mark.parametrize("accumulate", [False, True])
def test_put__dtypes(dtype, accumulate):
    utils.init_seed(0)
    numel = 64
    inp = _gen_typed((numel,), dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    # Duplicate indices only when accumulating: with `accumulate=False` a
    # repeated index is last-writer-wins and ATen does not specify the winner.
    count = numel * 4 if accumulate else numel // 2
    if accumulate:
        index = torch.randint(
            0, numel, (count,), dtype=torch.int64, device=flag_gems.device
        )
    else:
        index = torch.randperm(numel, device=flag_gems.device)[:count].to(torch.int64)
    source = _gen_typed((count,), dtype, flag_gems.device)

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_inp.put_(ref_index, ref_source, accumulate=accumulate)
    res = flag_gems.put_(inp, index, source, accumulate=accumulate)
    assert res is inp

    # Integer and bool results must match bit-exactly. Float and complex
    # accumulation goes through atomics, so the summation order varies between
    # runs and the low bits legitimately differ from the reference.
    if accumulate and (dtype.is_floating_point or dtype.is_complex):
        utils.gems_assert_close(inp, ref_inp, dtype, atol=accumulate_atol(dtype))
    else:
        utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.put_
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int16])
def test_put__narrow_int_accumulate_wraparound(dtype):
    # The staged-int32 accumulate path must wrap exactly like native's integer
    # add, so start near the dtype's maximum and overflow it.
    info = torch.iinfo(dtype)
    for init in [0, info.max - 3, info.min + 3]:
        inp = torch.full((4,), init, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp.clone())

        n = 20
        index = torch.zeros(n, dtype=torch.int64, device=flag_gems.device)
        source = torch.ones(n, dtype=dtype, device=flag_gems.device)

        ref_inp.put_(
            utils.to_reference(index), utils.to_reference(source), accumulate=True
        )
        flag_gems.put_(inp, index, source, accumulate=True)

        utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.put_
def test_put__bool_accumulate_is_logical_or():
    # Native `accumulate=True` on bool behaves as a logical OR: any True source
    # lands as True and an already-True slot stays True.
    for init in [False, True]:
        for k in [1, 2, 3, 257]:
            inp = torch.zeros(2, dtype=torch.bool, device=flag_gems.device)
            inp[0] = init
            ref_inp = utils.to_reference(inp.clone())

            index = torch.zeros(k, dtype=torch.int64, device=flag_gems.device)
            source = torch.ones(k, dtype=torch.bool, device=flag_gems.device)

            ref_inp.put_(
                utils.to_reference(index), utils.to_reference(source), accumulate=True
            )
            flag_gems.put_(inp, index, source, accumulate=True)

            utils.gems_assert_equal(inp, ref_inp)


# ---------------------------------------------------------------------------
# Arbitrary rank: `put_` addresses the flattened tensor, so rank is unbounded.
# ---------------------------------------------------------------------------
@pytest.mark.put_
@pytest.mark.parametrize("rank", [1, 2, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("contiguous", [True, False])
def test_put__high_rank(rank, contiguous):
    utils.init_seed(0)
    shape = tuple(range(2, 2 + rank))
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    if not contiguous:
        # A fully reversed permutation is the worst case for dim collapsing:
        # no two dimensions are memory-adjacent, so the rank survives intact.
        inp = inp.permute(*reversed(range(rank)))
    ref_inp = utils.to_reference(inp.clone())

    numel = inp.numel()
    count = max(1, numel // 3)
    index = torch.randperm(numel, device=flag_gems.device)[:count].to(torch.int64)
    source = torch.randn(count, dtype=torch.float32, device=flag_gems.device)

    ref_inp.put_(utils.to_reference(index), utils.to_reference(source))
    res = flag_gems.put_(inp, index, source)
    assert res is inp

    utils.gems_assert_equal(inp, ref_inp)


# ---------------------------------------------------------------------------
# Error semantics: these must match the exception *type* torch raises.
# ---------------------------------------------------------------------------
@pytest.mark.put_
def test_put__empty_self_raises():
    # Non-empty indices into a zero-element tensor: no valid index exists.
    inp = torch.zeros(0, dtype=torch.float32, device=flag_gems.device)
    index = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)
    source = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(IndexError):
        torch.zeros(0, dtype=torch.float32).put_(index.cpu(), source.cpu())
    with pytest.raises(IndexError):
        flag_gems.put_(inp, index, source)


@pytest.mark.put_
@pytest.mark.parametrize("self_numel", [0, 4])
def test_put__empty_index_is_noop(self_numel):
    # An empty index is valid even on an empty tensor, and writes nothing.
    inp = torch.randn(self_numel, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    index = torch.zeros(0, dtype=torch.int64, device=flag_gems.device)
    source = torch.zeros(0, dtype=torch.float32, device=flag_gems.device)

    ref_inp.put_(utils.to_reference(index), utils.to_reference(source))
    res = flag_gems.put_(inp, index, source)
    assert res is inp

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.put_
@pytest.mark.parametrize("bad", [4, 5, 100, -5, -100])
def test_put__index_out_of_range(bad):
    # ATen's CUDA `put_` enforces `-numel <= index < numel` with a *device-side*
    # assert, and our kernel mirrors it with an unconditional device trap rather
    # than `tl.device_assert` (which is compiled out unless TRITON_DEBUG=1).
    # A tripped device assert poisons the CUDA context for the whole process, so
    # the failure is exercised in an isolated subprocess.
    code = textwrap.dedent(f"""
        import torch
        import flag_gems

        inp = torch.zeros(4, dtype=torch.float32, device=flag_gems.device)
        index = torch.tensor([{bad}], dtype=torch.int64, device=flag_gems.device)
        source = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

        # A valid index must not synchronize on the success path.
        inp.put_(index[:0].new_zeros(0), source[:0])
        inp.put_(index, source)
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


@pytest.mark.put_
@pytest.mark.parametrize("boundary", [-4, -1, 0, 3])
def test_put__index_boundary_valid(boundary):
    # The values just inside the valid range must still be accepted.
    inp = torch.zeros(4, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    index = torch.tensor([boundary], dtype=torch.int64, device=flag_gems.device)
    source = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

    ref_inp.put_(utils.to_reference(index), utils.to_reference(source))
    flag_gems.put_(inp, index, source)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.put_
def test_put__device_mismatch_raises():
    # torch requires self/index/source on one device and refuses to transfer.
    inp = torch.zeros(4, dtype=torch.float32, device=flag_gems.device)
    index_gpu = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)
    source_gpu = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.put_(inp, index_gpu.cpu(), source_gpu)
    with pytest.raises(RuntimeError, match="same device"):
        flag_gems.put_(inp, index_gpu, source_gpu.cpu())


@pytest.mark.put_
def test_put__index_dtype_and_numel_raise():
    inp = torch.zeros(4, dtype=torch.float32, device=flag_gems.device)
    source = torch.ones(1, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="long tensor for index"):
        flag_gems.put_(
            inp,
            torch.tensor([0], dtype=torch.int32, device=flag_gems.device),
            source,
        )
    with pytest.raises(RuntimeError, match="same dtype"):
        flag_gems.put_(
            inp,
            torch.tensor([0], dtype=torch.int64, device=flag_gems.device),
            torch.ones(1, dtype=torch.float64, device=flag_gems.device),
        )
    with pytest.raises(IndexError, match="same number of elements"):
        flag_gems.put_(
            inp,
            torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device),
            source,
        )


# ---------------------------------------------------------------------------
# Memory-overlap semantics.
# ---------------------------------------------------------------------------
@pytest.mark.put_
def test_put__expanded_self_raises():
    # An expanded `self` aliases one element from several positions, so the
    # write target is ambiguous and torch refuses it.
    inp = torch.zeros(1, 4, dtype=torch.float32, device=flag_gems.device).expand(3, 4)
    index = torch.arange(4, dtype=torch.int64, device=flag_gems.device)
    source = torch.ones(4, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="more than one element"):
        torch.zeros(1, 4).expand(3, 4).put_(index.cpu(), source.cpu())
    with pytest.raises(RuntimeError, match="more than one element"):
        flag_gems.put_(inp, index, source)


@pytest.mark.put_
def test_put__aliased_index_or_source_raises():
    # torch rejects an index/source that overlaps `self`, since the write would
    # race against the read.
    index = torch.arange(4, dtype=torch.int64, device=flag_gems.device)

    inp = torch.arange(16, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="single memory location"):
        flag_gems.put_(inp, index, inp[4:8])

    # `source is self`: the index must cover all 16 elements, since the
    # index/source numel check runs before the overlap check.
    inp = torch.arange(16, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="single memory location"):
        flag_gems.put_(
            inp,
            torch.arange(16, dtype=torch.int64, device=flag_gems.device),
            inp,
        )

    idx_self = torch.arange(16, dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="single memory location"):
        flag_gems.put_(
            idx_self,
            idx_self[:4],
            torch.arange(100, 104, dtype=torch.int64, device=flag_gems.device),
        )


@pytest.mark.put_
def test_put__interleaved_views_allowed():
    # torch permits an overlap it classifies as "too hard" to check -- here
    # `self` and `source` interleave in one storage without sharing elements.
    # It must keep working and agree with native.
    base = torch.arange(20, dtype=torch.float32, device=flag_gems.device)
    ref_base = utils.to_reference(base.clone())

    index = torch.arange(10, dtype=torch.int64, device=flag_gems.device)
    ref_base[::2].put_(utils.to_reference(index), ref_base[1::2])
    flag_gems.put_(base[::2], index, base[1::2])

    utils.gems_assert_equal(base, ref_base)


@pytest.mark.put_
def test_put__disjoint_views_same_storage():
    # Non-overlapping slices of one storage are plainly legal.
    base = torch.arange(16, dtype=torch.float32, device=flag_gems.device)
    ref_base = utils.to_reference(base.clone())

    index = torch.arange(4, dtype=torch.int64, device=flag_gems.device)
    ref_base[0:4].put_(utils.to_reference(index), ref_base[8:12])
    flag_gems.put_(base[0:4], index, base[8:12])

    utils.gems_assert_equal(base, ref_base)


# ---------------------------------------------------------------------------
# Non-contiguous `index` / `source`.
# ---------------------------------------------------------------------------
# `index.reshape(-1)` is a no-op on an already-1-D strided view and preserves
# the non-unit stride, so the kernel must not assume unit-stride operands.
@pytest.mark.put_
@pytest.mark.parametrize("accumulate", [False, True])
def test_put__strided_source(accumulate):
    inp = torch.zeros(8, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    src_base = torch.arange(1, 17, dtype=torch.float32, device=flag_gems.device)
    source = src_base[::2]  # 1-D, stride 2
    index = torch.arange(8, dtype=torch.int64, device=flag_gems.device)

    ref_inp.put_(
        utils.to_reference(index), utils.to_reference(source), accumulate=accumulate
    )
    flag_gems.put_(inp, index, source, accumulate=accumulate)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.put_
@pytest.mark.parametrize("accumulate", [False, True])
def test_put__strided_index(accumulate):
    inp = torch.zeros(8, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    # Interleave the real indices with a sentinel so a unit-stride read would
    # pick up the wrong values instead of silently working.
    idx_base = torch.tensor(
        [0, 7, 1, 7, 2, 7, 3, 7, 4, 7, 5, 7, 6, 7, 7, 7],
        dtype=torch.int64,
        device=flag_gems.device,
    )
    index = idx_base[::2]  # 1-D, stride 2 -> 0..7
    source = torch.arange(1, 9, dtype=torch.float32, device=flag_gems.device)

    ref_inp.put_(
        utils.to_reference(index), utils.to_reference(source), accumulate=accumulate
    )
    flag_gems.put_(inp, index, source, accumulate=accumulate)

    utils.gems_assert_equal(inp, ref_inp)


@pytest.mark.put_
@pytest.mark.parametrize("accumulate", [False, True])
def test_put__transposed_index_and_source(accumulate):
    # A transposed 2-D index/source flattens in row-major order, which differs
    # from its memory order.
    inp = torch.zeros(6, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    index = torch.arange(6, dtype=torch.int64, device=flag_gems.device).reshape(2, 3).T
    source = (
        torch.arange(1, 7, dtype=torch.float32, device=flag_gems.device).reshape(2, 3).T
    )

    ref_inp.put_(
        utils.to_reference(index), utils.to_reference(source), accumulate=accumulate
    )
    flag_gems.put_(inp, index, source, accumulate=accumulate)

    utils.gems_assert_equal(inp, ref_inp)


# ---------------------------------------------------------------------------
# Views of a larger storage: writes must land in the view and leave the rest of
# the base tensor untouched.
# ---------------------------------------------------------------------------
@pytest.mark.put_
@pytest.mark.parametrize(
    "make_view, base_numel, index_vals",
    [
        (lambda b: b[5:15], 20, [0, 9]),
        (lambda b: b[::2], 20, [0, 4, 9]),
        (lambda b: b[3::2], 20, [0, 8]),
        (lambda b: b.view(4, 6)[1:3, 1:5], 24, [0, 3, 7]),
        (lambda b: b.view(4, 5).T, 20, [0, 5, 19]),
        (lambda b: b.flip(0), 20, [0, 19]),
        (lambda b: b.view(2, 3, 4).permute(2, 0, 1), 24, [0, 11, 23]),
    ],
)
def test_put__view_of_larger_storage(make_view, base_numel, index_vals):
    base = torch.arange(base_numel, dtype=torch.float32, device=flag_gems.device)
    ref_base = utils.to_reference(base.clone())

    index = torch.tensor(index_vals, dtype=torch.int64, device=flag_gems.device)
    source = -torch.arange(
        1, len(index_vals) + 1, dtype=torch.float32, device=flag_gems.device
    )

    make_view(ref_base).put_(utils.to_reference(index), utils.to_reference(source))
    view = make_view(base)
    res = flag_gems.put_(view, index, source)
    assert res is view

    # Comparing the whole base catches both a wrong offset and a stray write.
    utils.gems_assert_equal(base, ref_base)


@pytest.mark.put_
def test_put__complex32_rejected():
    # Native CUDA `put_` has no ComplexHalf kernel; `complex32` used to fall
    # through to the FP16 real-view path, silently succeeding where ATen raises.
    inp = torch.zeros(4, dtype=torch.complex32, device=flag_gems.device)
    index = torch.tensor([0], dtype=torch.int64, device=flag_gems.device)
    source = torch.ones(1, dtype=torch.complex32, device=flag_gems.device)

    with pytest.raises(NotImplementedError) as ref_err:
        inp.clone().put_(index, source)
    with pytest.raises(NotImplementedError) as res_err:
        flag_gems.put_(inp, index, source)

    assert str(ref_err.value) == str(res_err.value)
