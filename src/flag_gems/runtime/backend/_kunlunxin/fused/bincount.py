import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))

# ---------------------------------------------------------------------------
# Per-bin scalar-sequential kernels
#
# Design rationale (mirrors moe_align_block_size_stage1):
#   - grid = (output_size,), each program owns exactly ONE output bin
#   - Inner loop iterates over ALL input elements one by one (scalar loads)
#   - Conditional `if val == bin_id` accumulates only matching elements
#   - No atomics, no vectorised scatter, no tl.sum tree-reduction
#
# Why scalar-sequential matters for float:
#   Any parallel split of the input changes the fp32 intermediate totals and
#   makes the result diverge from torch.bincount's sequential scan by ~0.02
#   for n=100_000, which exceeds the test tolerance.  Scalar-sequential order
#   exactly reproduces torch.bincount's per-bin accumulation order, giving
#   bit-identical float results.
#
# XPU compatibility:
#   - `for i in range(n_elements)` with do_not_specialize is the same pattern
#     used by moe_align_block_size_stage1 on this backend.
#   - `if scalar_triton_bool:` inside a loop is likewise supported.
#   - isCloseUnrollControl=True prevents the compiler from trying to unroll
#     the dynamic-bound loop (which would blow up code size for large n).
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["n_elements"])
def _bincount_kernel(
    input_ptr,
    output_ptr,
    n_elements,
):
    """Integer bincount: count occurrences of each value (no weights)."""
    bin_id = tl.program_id(0).to(tl.int64)
    count = 0  # int32; consistent type throughout (no int64 tl.sum used)
    for i in range(n_elements):
        val = tl.load(input_ptr + i).to(tl.int64)
        if val == bin_id:
            count = count + 1
    tl.store(output_ptr + bin_id, count.to(tl.int64))


@triton.jit(do_not_specialize=["n_elements"])
def _bincount_weights_fp32_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
):
    """Weighted bincount with fp32 accumulation.

    Scalar-sequential order exactly matches torch.bincount's internal order,
    yielding bit-identical float32 results.
    """
    bin_id = tl.program_id(0).to(tl.int64)
    acc = 0.0  # float32 in Triton JIT (consistent with fp32 weight loads)
    for i in range(n_elements):
        val = tl.load(input_ptr + i).to(tl.int64)
        if val == bin_id:
            w = tl.load(weights_ptr + i).to(tl.float32)
            acc = acc + w
    tl.store(output_ptr + bin_id, acc)


@triton.jit(do_not_specialize=["n_elements"])
def _bincount_weights_fp64_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    n_elements,
):
    """Weighted bincount with fp64 accumulation."""
    bin_id = tl.program_id(0).to(tl.int64)
    # Explicit fp64 init to keep the loop-carried type consistent.
    acc = tl.zeros([1], dtype=tl.float64)[0]
    for i in range(n_elements):
        val = tl.load(input_ptr + i).to(tl.int64)
        if val == bin_id:
            w = tl.load(weights_ptr + i).to(tl.float64)
            acc = acc + w
    tl.store(output_ptr + bin_id, acc)


def bincount(input, weights=None, minlength=0):
    logger.debug("GEMS_KUNLUNXIN BINCOUNT")

    assert input.dim() == 1, "input must be a 1-D tensor"
    assert minlength >= 0, "minlength must be non-negative"

    if weights is not None:
        assert weights.shape == input.shape, "weights must have the same shape as input"

    n = input.numel()

    if n == 0:
        if weights is not None:
            return torch.zeros(minlength, dtype=weights.dtype, device=input.device)
        return torch.zeros(minlength, dtype=torch.int64, device=input.device)

    input_contig = input.contiguous()

    # Determine output size; use PyTorch max to avoid tl.atomic_max with int64
    # (incomplete support on XPU).
    max_val = int(input_contig.max().item())
    output_size = max(max_val + 1, minlength)

    grid = (output_size,)

    if weights is None:
        output = torch.zeros(output_size, dtype=torch.int64, device=input.device)
        _bincount_kernel[grid](
            input_contig,
            output,
            n,
            isCloseUnrollControl=True,
        )
        return output

    weights_contig = weights.contiguous()
    out_dtype = weights.dtype

    if out_dtype == torch.float64:
        output = torch.zeros(output_size, dtype=torch.float64, device=input.device)
        _bincount_weights_fp64_kernel[grid](
            input_contig,
            weights_contig,
            output,
            n,
            isCloseUnrollControl=True,
        )
    else:
        # Accumulate in fp32 for fp16 / bf16 / fp32 weights
        output = torch.zeros(output_size, dtype=torch.float32, device=input.device)
        _bincount_weights_fp32_kernel[grid](
            input_contig,
            weights_contig,
            output,
            n,
            isCloseUnrollControl=True,
        )
        if out_dtype != torch.float32:
            output = output.to(out_dtype)

    return output
