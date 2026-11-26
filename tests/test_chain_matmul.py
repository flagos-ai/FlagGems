# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.fixture(autouse=True)
def _fp32_reference_ieee():
    # The aten fp32 reference must run at true fp32 (ieee), matching the
    # candidate kernels' input_precision="ieee". On some runners cuBLAS TF32
    # is enabled globally, which leaves the reference at ~1e-4 relative error
    # on chained matmuls while the candidate stays at ~1e-7 — the comparison
    # against the fp32 tolerance (1.3e-6) then fails on whichever random shape
    # draws a long reduction. Pin both precision APIs off (torch >= 2.9 new
    # API and the legacy flag, which older versions only have) and restore
    # whatever the environment had afterwards.
    saved_precision = None
    try:
        saved_precision = torch.backends.cuda.matmul.fp32_precision
    except (AttributeError, RuntimeError):
        pass
    saved_tf32 = None
    try:
        saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    except RuntimeError:
        pass
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    except AttributeError:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    if saved_precision is not None:
        try:
            torch.backends.cuda.matmul.fp32_precision = saved_precision
        except RuntimeError:
            pass
    if saved_tf32 is not None:
        try:
            torch.backends.cuda.matmul.allow_tf32 = saved_tf32
        except RuntimeError:
            pass


# The helpers below (value-range resolution, native-dtype comparison, the
# special-value payloads) come from the KernelGen test harness that the kernel
# was evaluated against. That harness lives outside this repository, so the
# parts this operator depends on are inlined here rather than added as a new
# shared module.

# The five value ranges of the operator-test spec: [-1,1], [0,1], [-1,0],
# [0,dtype_max], [dtype_min,0]. The extreme ones matter for a matmul chain
# because the intermediate products are what overflow first.
VALUE_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]


def _resolve_bound(symbol, dtype):
    finfo = torch.finfo(dtype)
    return {
        "-1": -1.0,
        "0": 0.0,
        "1": 1.0,
        "max": finfo.max,
        "min": finfo.min,
    }[symbol]


def _make_input(dtype, shape, value_range, scale=False):
    low = _resolve_bound(value_range[0], dtype)
    high = _resolve_bound(value_range[1], dtype)
    if low == high:
        return torch.full(shape, low, device=flag_gems.device, dtype=dtype)
    tensor = torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )
    if scale:
        # Normalize by sqrt(K) so the chain product keeps the magnitude of its
        # inputs instead of growing by a factor of ~sqrt(K) per matmul. The
        # long chains below reduce over extents up to 2048, which without this
        # drives the product of unit-range inputs to ~1e5 -- past the fp16 max
        # of 65504, so both the reference and the candidate saturate to Inf and
        # the case stops testing the kernel at all. Scaling in fp32 before the
        # cast keeps the small-dtype inputs from losing bits to the division.
        tensor = (tensor.to(torch.float32) / math.sqrt(shape[0])).to(dtype)
    return tensor


def _make_chain_inputs(dtype, shapes, value_range):
    # Only the interior factors are normalized: scaling the first matrix would
    # shrink the result without shortening the reduction it is scaled for.
    return [
        _make_input(dtype, shape, value_range, scale=bool(index))
        for index, shape in enumerate(shapes)
    ]


def _reduce_dim(shapes):
    # Largest extent in the chain, matching how the other chained-matmul test
    # (test_linalg_multi_dot.py) scales its tolerance.
    return max(extent for shape in shapes for extent in shape)


def _assert_close(res, ref, reduce_dim):
    # The reference is kept in the input dtype, so each DP node rounds the same
    # way the candidate does; comparing against an upcast reference would
    # instead measure the chain's accumulated rounding, which differs with the
    # parenthesization rather than with correctness.
    #
    # The tolerance still has to grow with the chain's reduction extent. Both
    # sides round per node, but they do not sum each node in the same order:
    # the DP picks the parenthesization while the tile shape, split-K factor
    # and K-step order of each gemm are chosen per shape and per device, so the
    # candidate's reduction order is its own. Over a long reduction those
    # orders diverge by a few ulp of the accumulator, which is exactly what
    # reduce_dim is for.
    #
    # Overflow cases (the [0,max] / [min,0] ranges over long chains) need a
    # looser contract: whether an overflowed intermediate poisons the output as
    # inf or nan depends on where the accumulation saturates, which differs
    # between the CPU reference's BLAS path and any CUDA kernel even when both
    # sides compute the same parenthesization. Both sides are then required to
    # be non-finite at the same positions and close wherever both are finite;
    # the inf/nan flavor of an overflowed element is not part of the operator's
    # semantics.
    res = res.cpu()
    ref = ref.cpu()
    # The fp32 upcast only classifies the overflow mask; the element comparison
    # itself stays in the original dtype so its tolerance applies unchanged.
    mask_res = torch.isfinite(res.float())
    mask_ref = torch.isfinite(ref.float())
    if (mask_res & mask_ref).all():
        utils.gems_assert_close(
            res, ref, ref.dtype, equal_nan=True, reduce_dim=reduce_dim
        )
        return
    assert (mask_res == mask_ref).all(), (
        "overflow positions differ: "
        f"res non-finite {int((~mask_res).sum())}, "
        f"ref non-finite {int((~mask_ref).sum())}"
    )
    if mask_res.any():
        # Both sides overflow at the same positions; compare the remaining
        # finite elements, where input nan (equal_nan=True) still applies.
        utils.gems_assert_close(
            res[mask_res],
            ref[mask_ref],
            ref.dtype,
            equal_nan=True,
            reduce_dim=reduce_dim,
        )


def _to_reference(inp, device=None):
    # Copy dense storage instead of cloning, so stride/offset of a transposed
    # view survive into the reference.
    reference = torch.empty(0, dtype=inp.dtype, device=inp.device).set_(
        inp.untyped_storage().clone(),
        inp.storage_offset(),
        inp.size(),
        inp.stride(),
    )
    reference.requires_grad_(inp.requires_grad)
    reference = utils.to_reference(reference)
    if device is not None:
        reference = reference.to(device)
    return reference


# Fixed payloads so NaN / Inf land at known positions of the first matrix.
SPECIAL_PAYLOADS = {
    "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
    "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
    "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
}

SPECIAL_CASES = [
    (dtype, scenario)
    for dtype in utils.ALL_FLOAT_DTYPES
    for scenario in SPECIAL_PAYLOADS
]

# The six large chains of the core benchmark. They are the shapes the kernel
# config table is tuned for, so the accuracy tests must cover them too.
LARGE_CHAIN_DIMENSIONS = [
    (1024, 64, 1024),
    (2048, 32, 2048, 32),
    (512, 2048, 64, 1024),
    (1024, 128, 2048, 64, 512),
    (256, 1024, 64, 2048, 128, 512),
    (512, 128, 1024, 64, 2048, 32, 512),
]

# Each case is one chain: a list of rank-2 shapes with consistent inner
# dimensions. The small chains cover a one-matrix chain, the degenerate
# single-element reduction, the odd / non-power-of-two extents that exercise
# the masked tails of the gemm kernels, and chains long enough that the
# matrix-chain DP picks a non-trivial parenthesization.
CHAIN_SHAPES = [
    [(1, 1)],
    [(4, 8)],
    [(2, 3), (3, 4)],
    [(1, 5), (5, 1), (1, 7)],
    [(16, 32), (32, 64), (64, 32), (32, 16)],
    [(8, 16), (16, 32), (32, 48), (48, 32), (32, 16)],
    [(33, 65), (65, 17), (17, 129), (129, 255), (255, 71)],
] + [list(zip(dims, dims[1:])) for dims in LARGE_CHAIN_DIMENSIONS]

# A subset for the .out overload: it measures the same chains plus the write
# into the caller's buffer, so the long chains add no new path.
OUT_CHAIN_SHAPES = CHAIN_SHAPES[:4]

BACKWARD_CHAINS = [
    [(4, 8)],
    [(2, 3), (3, 4)],
    [(4, 8), (8, 16), (16, 4)],
    [(16, 32), (32, 64), (64, 32), (32, 16)],
]

NONCONTIG_CHAINS = [
    [(4, 8), (8, 16)],
    [(16, 32), (32, 64), (64, 16)],
]


def _ref_device(dtype, shapes):
    # Device for the aten reference under --ref cpu. For chains of >= 3
    # low-precision matrices the CPU reference is not a usable oracle: each
    # node rounds to fp16/bf16, so the CPU path's and any GPU path's per-node
    # roundings accumulate in their own reduction orders — aten's own CUDA
    # result exceeds this test's tolerance against its own CPU result by up
    # to ~500x on the six-matrix chains. The same-device native op stays the
    # oracle there (the fp32/fp64 references remain on CPU as the lane
    # intends; two-matrix chains agree cross-device within tolerance).
    if utils.TO_CPU and len(shapes) > 2 and dtype in (torch.float16, torch.bfloat16):
        return flag_gems.device
    return None


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", CHAIN_SHAPES)
@pytest.mark.parametrize("value_range", VALUE_RANGES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_chain_matmul(shapes, value_range, dtype):
    inp = _make_chain_inputs(dtype, shapes, value_range)
    dev = _ref_device(dtype, shapes)
    ref_inp = [_to_reference(matrix, dev) for matrix in inp]

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    res_out = flag_gems.chain_matmul(inp)

    assert res_out.is_contiguous()
    _assert_close(res_out, ref_out.to(dtype), _reduce_dim(shapes))


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", NONCONTIG_CHAINS)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_chain_matmul_non_contiguous(shapes, dtype):
    # Transposed views: the kernels are stride-generic and must not assume a
    # row-major layout.
    inp = [_make_input(dtype, (cols, rows), ["-1", "1"]).t() for rows, cols in shapes]
    assert all(not matrix.is_contiguous() for matrix in inp)
    ref_inp = [_to_reference(matrix, _ref_device(dtype, shapes)) for matrix in inp]

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    res_out = flag_gems.chain_matmul(inp)

    _assert_close(res_out, ref_out.to(dtype), _reduce_dim(shapes))


@pytest.mark.chain_matmul_out
@pytest.mark.parametrize("shapes", OUT_CHAIN_SHAPES)
@pytest.mark.parametrize("value_range", VALUE_RANGES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
def test_chain_matmul_out(shapes, value_range, dtype):
    inp = _make_chain_inputs(dtype, shapes, value_range)
    dev = _ref_device(dtype, shapes)
    ref_inp = [_to_reference(matrix, dev) for matrix in inp]

    out_shape = (shapes[0][0], shapes[-1][1])
    # Garbage-prefilled buffers: the .out overload must overwrite every element.
    out = torch.full(out_shape, 7.0, dtype=dtype, device=flag_gems.device)
    ref_out = torch.full(
        out_shape, 7.0, dtype=ref_inp[0].dtype, device=ref_inp[0].device
    )

    torch.ops.aten.chain_matmul.out(ref_inp, out=ref_out)
    res_ret = flag_gems.chain_matmul(inp, out=out)

    # The .out overload must write into and return the caller's buffer.
    assert res_ret is out
    _assert_close(out, ref_out.to(dtype), _reduce_dim(shapes))


@pytest.mark.chain_matmul
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_chain_matmul_nan_inf(dtype, scenario):
    # Zeros in m2 also exercise Inf * 0 -> NaN during reduction.
    m1 = (
        torch.tensor(
            SPECIAL_PAYLOADS[scenario], device=flag_gems.device, dtype=torch.float32
        )
        .to(dtype)[:4]
        .reshape(2, 2)
    )
    m2 = torch.tensor(
        [[1.0, 0.0], [1.0, 1.0]],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp = [m1, m2]
    ref_inp = [_to_reference(matrix) for matrix in inp]

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    res_out = flag_gems.chain_matmul(inp)

    _assert_close(res_out, ref_out, _reduce_dim([m1.shape, m2.shape]))


@pytest.mark.chain_matmul
@pytest.mark.parametrize("shapes", BACKWARD_CHAINS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_chain_matmul_backward(shapes, dtype):
    # The backward of a multi-matrix chain feeds each gradient through several
    # chained matmuls; in the low dtypes each one rounds to the input dtype, so
    # a gradient element that lands near a rounding boundary can differ by a
    # couple of ulp between any two implementations depending on their
    # reduction order. Unseeded inputs make that boundary crossing a coin flip
    # per run, so fix the seed (repo precedent: test_ctc_loss, test_dirichlet_grad).
    utils.init_seed(14)
    inp = [_make_input(dtype, shape, ["-1", "1"]).requires_grad_() for shape in shapes]
    grad = _make_input(dtype, (shapes[0][0], shapes[-1][1]), ["-1", "1"])

    dev = _ref_device(dtype, shapes)
    ref_inp = [
        _to_reference(matrix.detach(), dev).requires_grad_(True) for matrix in inp
    ]
    ref_grad = _to_reference(grad, dev)

    ref_out = torch.ops.aten.chain_matmul(ref_inp)
    ref_grads = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad)

    reduce_dim = _reduce_dim(shapes)

    # The candidate forward must match the native-dtype reference...
    res_out = flag_gems.chain_matmul(inp)
    _assert_close(res_out, ref_out.to(dtype), reduce_dim)

    # ...and so must every input gradient, with the dtype preserved.
    assert res_out.requires_grad
    res_grads = torch.autograd.grad(res_out, inp, grad_outputs=grad)
    for res_grad, ref_g in zip(res_grads, ref_grads):
        assert res_grad.dtype == dtype
        assert res_grad.shape == ref_g.shape
        _assert_close(res_grad, ref_g.to(dtype), reduce_dim)


@pytest.mark.chain_matmul
def test_chain_matmul_rejects_empty_list():
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul([])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul([])


@pytest.mark.chain_matmul
def test_chain_matmul_rejects_non_tensor():
    with pytest.raises((RuntimeError, TypeError)):
        torch.ops.aten.chain_matmul(3.14)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(3.14)


@pytest.mark.chain_matmul
def test_chain_matmul_rejects_1d_matrix():
    inp = [_make_input(torch.float32, (4,), ["-1", "1"])]
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(inp)


@pytest.mark.chain_matmul
def test_chain_matmul_rejects_3d_tensor():
    matrix = _make_input(torch.float32, (2, 3, 4), ["-1", "1"])
    inp = [matrix, matrix]
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(inp)


@pytest.mark.chain_matmul
def test_chain_matmul_rejects_mismatched_dims():
    inp = [
        _make_input(torch.float32, (3, 4), ["-1", "1"]),
        _make_input(torch.float32, (5, 6), ["-1", "1"]),
    ]
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(inp)


@pytest.mark.chain_matmul
@pytest.mark.skipif(
    flag_gems.device == "cpu",
    reason="aten chain_matmul accepts integer addmm on the CPU reference path",
)
def test_chain_matmul_rejects_int_dtype():
    inp = [torch.ones((2, 2), dtype=torch.int32, device=flag_gems.device)] * 2
    # On the accelerator the integer addmm is not implemented.
    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul(inp)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(inp)


@pytest.mark.chain_matmul_out
def test_chain_matmul_out_rejects_wrong_dtype():
    inp = [_make_input(torch.float32, shape, ["-1", "1"]) for shape in [(4, 8), (8, 4)]]
    ref_inp = [_to_reference(matrix) for matrix in inp]

    ref_bad = torch.empty(4, 4, dtype=torch.int32, device=ref_inp[0].device)
    res_bad = torch.empty(4, 4, dtype=torch.int32, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        torch.ops.aten.chain_matmul.out(ref_inp, out=ref_bad)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems.chain_matmul(inp, out=res_bad)
