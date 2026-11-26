# SPDX-License-Identifier: Apache-2.0
"""Build step-scoped MACA prefill query/KV offsets without a host readback."""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _prepare_maca_prefill_metadata_kernel(
    Seq,
    Query,
    OutQuery,
    OutKV,
    seq_stride: tl.constexpr,
    query_stride: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    # Include the leading zero in the scan, avoiding a separate memset/cat.
    lens = tl.load(Seq + (i - 1) * seq_stride, (i > 0) & (i <= N), 0)
    kv_offsets = tl.cumsum(lens.to(tl.int32), 0)
    query_base = tl.load(Query)
    query_offsets = tl.load(Query + i * query_stride, i <= N, 0) - query_base
    tl.store(OutQuery + i, query_offsets, i <= N)
    tl.store(OutKV + i, kv_offsets, i <= N)


def prepare_maca_prefill_metadata(seq_lens, query_starts):
    """Return normalized query offsets and leading-zero int32 KV offsets.

    ``seq_lens`` describes only the prefill suffix of the current batch;
    ``query_starts`` is its n+1-element slice of the full query offset array.
    Outputs belong to this call. No tensor values are cached across steps.
    The bounded single-CTA scan targets small attention metadata arrays;
    larger arrays and other devices/dtypes keep a device-side PyTorch path.
    """
    if seq_lens.ndim != 1 or query_starts.ndim != 1:
        raise ValueError("Expected one-dimensional metadata tensors")
    n = seq_lens.numel()
    if query_starts.numel() != n + 1 or query_starts.device != seq_lens.device:
        raise ValueError("Expected n+1 query offsets on the sequence-length device")
    if (
        seq_lens.device.type != "cuda"
        or seq_lens.dtype != torch.int32
        or query_starts.dtype != torch.int32
        or n > 4095
    ):
        kv = torch.empty(n + 1, dtype=torch.int32, device=seq_lens.device)
        kv[0] = 0
        torch.cumsum(seq_lens, 0, dtype=torch.int32, out=kv[1:])
        return query_starts - query_starts[0], kv

    query = torch.empty(n + 1, dtype=torch.int32, device=seq_lens.device)
    kv = torch.empty_like(query)
    block = max(64, triton.next_power_of_2(n + 1))
    # C500 uses Wave64. Keep small metadata to one wave and avoid deep staging.
    with torch_device_fn.device(seq_lens.device):
        _prepare_maca_prefill_metadata_kernel[(1,)](
            seq_lens,
            query_starts,
            query,
            kv,
            seq_lens.stride(0),
            query_starts.stride(0),
            n,
            block,
            num_warps=1 if block < 512 else 4,
            num_stages=1,
        )
    return query, kv
