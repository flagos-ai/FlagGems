import functools
import inspect
import logging
import os
import time

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

TOTAL_CORE_NUM = tl.constexpr(16)
_ROW_BLOCK = 256


def _log_enabled():
    return os.getenv("FLAG_GEMS_LOG_INDEX_SELECT", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _fmt_value(value):
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, "
            f"stride={tuple(value.stride())}, dtype={value.dtype})"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_value(v) for v in value) + "]"
    return repr(value)


def log_args(func):
    """Print each argument on one line and the wall-clock execution time in ms.

    Example output:
        [index_select] inp: Tensor(shape=(2046, 2176), stride=(2176, 1), dtype=torch.int32)
        [index_select] dim: 0
        [index_select] index: Tensor(shape=(4,), stride=(1,), dtype=torch.int64)
        [index_select] elapsed_ms=12.345

    grep examples:
        grep '\\[index_select\\]' log.txt                     # all lines of this call
        grep '\\[index_select\\] elapsed_ms' log.txt          # only the timing lines
        grep '\\[index_select\\] inp:' log.txt                # only the `inp` argument
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _log_enabled():
            return func(*args, **kwargs)

        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        prefix = f"[{func.__name__}]"
        for name, value in bound.arguments.items():
            print(f"{prefix} {name}: {_fmt_value(value)}", flush=True)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"{prefix} elapsed_ms={elapsed_ms:.3f}", flush=True)

    return wrapper


def dim0_rows_are_dense(inp: torch.Tensor) -> bool:
    """True when each inp[i, ...] is a contiguous tail block (dim0 stride may differ).

    Covers contiguous N-D tensors and dense-row views such as parent[:, 0:4, :, :],
    where is_contiguous() is False but stride(1:) still matches a packed row.
    """
    if inp.ndim < 2:
        # A 1-D "row" is a single element, so it is trivially a dense tail
        # block; the caller's kernel then uses stride(0) as the element stride.
        # (ndim == 0 cannot reach here: index_select validates dim first.)
        return inp.ndim == 1
    stride = inp.stride()
    shape = inp.shape
    step = 1
    for d in range(inp.ndim - 1, 0, -1):
        if stride[d] != step:
            return False
        step *= shape[d]
    return True


@libentry()
@triton.jit(do_not_specialize=["index_len", "M", "row_stride"])
def index_select_dim0_kernel(
    out_ptr,
    index_ptr,
    inp_ptr,
    index_len,
    M,
    row_stride,
    BLOCK: tl.constexpr,
):
    # 16-CTA (= TX81 cores): each owns an output-row slice, tiles along tail dim.
    pid = tle.program_id(0)
    rows_per_tile = tl.cdiv(index_len, TOTAL_CORE_NUM)
    start = pid * rows_per_tile
    end = min(start + rows_per_tile, index_len)

    for row in range(start, end):
        row_idx = tl.load(index_ptr + row).to(tl.int64)
        src_base = row_idx * tl.cast(row_stride, tl.int64)
        dst_base = tl.cast(row, tl.int64) * tl.cast(M, tl.int64)
        for blk in range(0, tl.cdiv(M, BLOCK)):
            cols = (blk * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
            mask = cols < M
            vals = tl.load(inp_ptr + src_base + cols, mask=mask, other=0.0)
            tl.store(out_ptr + dst_base + cols, vals, mask=mask)


@libentry()
@triton.jit(
    do_not_specialize=[
        "index_len",
        "hidden_size",
        "state_width",
        "inp_stride0",
        "inp_stride1",
        "inp_stride2",
    ]
)
def index_select_dim0_strided_3d_kernel(
    out_ptr,
    index_ptr,
    inp_ptr,
    index_len,
    hidden_size,
    state_width,
    inp_stride0,
    inp_stride1,
    inp_stride2,
    BLOCK: tl.constexpr,
):
    """Gather rank-3 dim-0 rows using the input tensor's actual strides."""
    pid = tle.program_id(0)
    rows_per_tile = tl.cdiv(index_len, TOTAL_CORE_NUM)
    start = pid * rows_per_tile
    end = min(start + rows_per_tile, index_len)
    row_elements = hidden_size * state_width

    for row in range(start, end):
        row_idx = tl.load(index_ptr + row).to(tl.int64)
        src_row = row_idx * tl.cast(inp_stride0, tl.int64)
        dst_row = tl.cast(row, tl.int64) * tl.cast(row_elements, tl.int64)
        for blk in range(0, tl.cdiv(row_elements, BLOCK)):
            offsets = blk * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < row_elements
            hidden = offsets // state_width
            width = offsets % state_width
            src = (
                inp_ptr
                + src_row
                + hidden.to(tl.int64) * tl.cast(inp_stride1, tl.int64)
                + width.to(tl.int64) * tl.cast(inp_stride2, tl.int64)
            )
            values = tl.load(src, mask=mask, other=0.0)
            tl.store(out_ptr + dst_row + offsets, values, mask=mask)


def _use_fast_dim0(inp: torch.Tensor, dim: int) -> bool:
    """Any dim0 gather whose rows are dense in memory (no per-shape whitelist)."""
    if dim % inp.ndim != 0:
        return False
    return dim0_rows_are_dense(inp)


def _fallback_cpu(inp, dim, index):
    """Mirror torch_txda PrivateUse1 index_select backend fallback."""
    if _log_enabled():
        print(
            "TXDA index_select CPU fallback:",
            f"shape={tuple(inp.shape)}",
            f"stride={inp.stride()}",
            f"dim={dim}",
            flush=True,
        )
    device = inp.device
    out = torch.index_select(inp.cpu(), dim, index.cpu())
    return out.to(device=device, non_blocking=False)


@log_args
def index_select(inp, dim, index):
    logger.debug("GEMS_TSINGMICRO INDEX_SELECT")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert index.ndim <= 1, "Index should have dimension 1 or 0"

    dim = dim % inp.ndim
    use_dense_rows = _use_fast_dim0(inp, dim)
    # Qwen PLE exposes physical [rows, width, hidden] storage as a logical
    # [rows, hidden, width] transpose. Its inner dimensions are not dense, but
    # each source element remains addressable from the tensor's three strides.
    use_strided_3d = dim == 0 and inp.ndim == 3
    if not use_dense_rows and not use_strided_3d:
        return _fallback_cpu(inp, dim, index)

    if index.ndim == 0:
        index = index.unsqueeze(0)
    index = index.contiguous()
    index_len = index.numel()

    if inp.shape[0] == 0:
        if index_len == 0:
            # No rows to gather: return the empty result instead of dividing
            # by zero below (tail_numel = numel // shape[0]).
            return torch.empty(
                (index_len, *inp.shape[1:]), dtype=inp.dtype, device=inp.device
            )
        # A non-empty index on an empty dim-0 is out of bounds; let the exact
        # fallback raise PyTorch's own error (IndexError for 1-D input,
        # RuntimeError for a multi-dim empty tensor).
        return _fallback_cpu(inp, dim, index)

    tail_numel = inp.numel() // inp.shape[0]
    out = torch.empty((index_len, *inp.shape[1:]), dtype=inp.dtype, device=inp.device)

    with torch_device_fn.device(inp.device):
        if use_dense_rows:
            index_select_dim0_kernel[(TOTAL_CORE_NUM,)](
                out,
                index,
                inp,
                index_len,
                tail_numel,
                inp.stride(0),
                BLOCK=_ROW_BLOCK,
                num_warps=1,
                num_stages=1,
            )
        else:
            index_select_dim0_strided_3d_kernel[(TOTAL_CORE_NUM,)](
                out,
                index,
                inp,
                index_len,
                inp.shape[1],
                inp.shape[2],
                inp.stride(0),
                inp.stride(1),
                inp.stride(2),
                BLOCK=_ROW_BLOCK,
                num_warps=1,
                num_stages=1,
            )
    return out
