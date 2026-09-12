import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes covering 1-D, multi-D and large flatten spans. Mirrors `test_put_.py`.
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

    if accumulate:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=accumulate_atol(dtype))
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


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
    source = torch.randn(count, dtype=dtype, device=flag_gems.device)

    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source)
    res_out = flag_gems.put(inp, index, source)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.put
@pytest.mark.parametrize("shape", [(64, 64), (20, 320, 15), (16, 128, 64, 60)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_put_non_contiguous(shape, dtype):
    # A transposed (non-contiguous) tensor exercises the multi-dim offset path.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = inp.transpose(0, 1)
    ref_inp = utils.to_reference(inp)

    index, source = gen_index_and_source(inp, dtype, flag_gems.device, accumulate=False)
    ref_index = utils.to_reference(index)
    ref_source = utils.to_reference(source)

    ref_out = torch.put(ref_inp, ref_index, ref_source)
    res_out = flag_gems.put(inp, index, source)

    utils.gems_assert_close(res_out, ref_out, dtype)


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
    source = torch.randn(
        (count // 4, 4) if count >= 4 else (count,),
        dtype=dtype,
        device=flag_gems.device,
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
    # The `put.out` reference uses `torch.ops.aten.put.out` (torch.put has no
    # `out=` overload). put is device-independent, so CPU/GPU references are
    # numerically identical; skip only under --ref=cpu where the aten.out
    # reference is forced onto CPU while the GEMS kernel stays on GPU.
    if utils.TO_CPU:
        pytest.skip("put.out uses aten.put.out reference; skip under --ref=cpu")

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

    if accumulate:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=accumulate_atol(dtype))
        utils.gems_assert_close(out, ref_out, dtype, atol=accumulate_atol(dtype))
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)
        utils.gems_assert_close(out, ref_out, dtype)
