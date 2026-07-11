import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


# ── dense kernel: single pass, writes all class values per row ────────────────


@libentry()
@triton.jit
def one_hot_dense_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Dense one-hot kernel.

    Each program processes BLOCK_M rows.  For each row it writes all
    ``num_classes`` values (only one of which is 1).  When ``num_classes``
    is larger than BLOCK_N the column dimension is tiled with a loop.
    """
    pid = ext.program_id(axis=0)
    row_start = pid * BLOCK_M
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < num_elements

    indices = tl.load(input_ptr + row_offsets, mask=row_mask, other=0)

    for col_st in range(0, num_classes, BLOCK_N):
        col_offsets = col_st + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < num_classes

        result = indices[:, None] == col_offsets[None, :]
        result = result.to(tl.int64)

        out_offsets = row_offsets[:, None] * num_classes + col_offsets[None, :]
        full_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(output_ptr + out_offsets, result, mask=full_mask)


# ── scatter kernel: only write the "1" positions (output must be zeroed first) ──


@libentry()
@triton.jit
def one_hot_scatter_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    num_classes,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter one-hot kernel.

    For each input element it computes ``row * num_classes + index`` and
    stores a single ``1`` at that position.  The output buffer must have
    been zero-initialized *before* this kernel is launched.
    """
    pid = ext.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    indices = tl.load(input_ptr + offsets, mask=mask, other=0)
    out_offsets = offsets * num_classes + indices
    tl.store(output_ptr + out_offsets, 1, mask=mask)


# ── block-size helpers ─────────────────────────────────────────────────────────


def _dense_block_size(num_elements: int, num_classes: int) -> int:
    """Pick an optimal BLOCK_M for the dense kernel.

    Balances per-block output against L2 cache budget while keeping enough
    grid blocks to utilise all multiprocessors.
    """
    per_row_bytes = num_classes * 8  # int64
    target_block_bytes = 65536  # 64 KB  (most L1/L2 will hold this easily)
    max_rows = max(32, target_block_bytes // per_row_bytes)
    max_rows = 1 << (max_rows.bit_length() - 1)

    if num_elements <= 4096:
        base = 512
    elif num_elements <= 65536:
        base = 1024
    else:
        base = 2048

    bs = min(base, max_rows)
    # At least 2 grid blocks → enough parallelism
    upper = max(256, triton.cdiv(num_elements, 2))
    bs = min(bs, upper)
    bs = 1 << (bs.bit_length() - 1)
    return bs


def _scatter_block_size(num_elements: int) -> int:
    """Pick an optimal BLOCK_SIZE for the scatter kernel."""
    if num_elements <= 1024:
        return 256
    elif num_elements <= 16384:
        return 512
    else:
        return 1024


# When num_classes exceeds this threshold the scatter path is used instead
# of the dense path because the per-element memory traffic of the dense
# kernel (writes  all  class values) becomes more expensive than zero-init
# + scatter (writes only the "1" positions).
_DENSE_THRESHOLD = 1024


# ── main entry point ───────────────────────────────────────────────────────────


def one_hot(tensor: torch.Tensor, num_classes: int = -1) -> torch.Tensor:
    logger.debug("GEMS_MTHREADS ONE_HOT")

    if tensor.dtype != torch.int64:
        raise RuntimeError(
            "one_hot is only applicable to index tensor of type LongTensor."
        )

    if tensor.numel() == 0:
        if num_classes <= 0:
            raise RuntimeError(
                "Can not infer total number of classes from empty tensor."
            )
        shape = (*tensor.shape, num_classes)
        return torch.empty(shape, device=tensor.device, dtype=torch.int64)

    # Infer num_classes from data when required (single device → host sync).
    if num_classes == -1:
        maxv = int(tensor.max().item())
        num_classes = maxv + 1
        if (tensor < 0).any():
            raise RuntimeError("Class values must be non-negative.")
    else:
        invalid = (tensor < 0) | (tensor >= num_classes)
        if invalid.any():
            if (tensor < 0).any():
                raise RuntimeError("Class values must be non-negative.")
            else:
                raise RuntimeError("Class values must be smaller than num_classes.")

    if num_classes < 1:
        raise RuntimeError("num_classes should be positive")

    if tensor.device.type == "cpu":
        out = torch.zeros((*tensor.shape, num_classes), device="cpu", dtype=torch.int64)
        out.scatter_(-1, tensor.unsqueeze(-1), 1)
        return out

    flat_input = tensor.contiguous().view(-1)
    num_elements = flat_input.numel()

    with torch_device_fn.device(tensor.device):
        if num_classes <= _DENSE_THRESHOLD:
            # Dense approach: single pass, write all class values.
            BLOCK_N = min(triton.next_power_of_2(num_classes), 128)
            BLOCK_M = _dense_block_size(num_elements, num_classes)
            out = torch.empty(
                num_elements * num_classes,
                device=tensor.device,
                dtype=torch.int64,
            )
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_M"]),)
            one_hot_dense_kernel[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
        else:
            # Scatter approach: zero-init the output then write only the "1"
            # positions.  This saves bandwidth when num_classes is large.
            out = torch.zeros(
                num_elements * num_classes,
                device=tensor.device,
                dtype=torch.int64,
            )
            BLOCK_SIZE = _scatter_block_size(num_elements)
            grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)
            one_hot_scatter_kernel[grid](
                flat_input,
                out,
                num_elements,
                num_classes,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    return out.view(*tensor.shape, num_classes)
