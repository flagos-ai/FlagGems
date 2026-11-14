# Copyright 2026 FlagOS Contributors
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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.nonzero import nonzero as _generic_nonzero
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
_BLOCK = 1024
_MAX_BLOCKS = 4096


@libentry()
@triton.jit
def _count_nonzero(inp, work, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(inp + offsets, mask=offsets < N, other=0)
    selected = (offsets < N) & (values != 0)
    tl.store(work + pid, tl.sum(selected.to(tl.int32), axis=0))


@libentry()
@triton.jit
def _scan_counts(work, total, NB, SCAN: tl.constexpr):
    offsets = tl.arange(0, SCAN)
    counts = tl.load(work + offsets, mask=offsets < NB, other=0)
    inclusive = tl.cumsum(counts, axis=0)
    tl.store(work + NB + offsets, inclusive - counts, mask=offsets < NB)
    tl.store(total, tl.sum(counts, axis=0))


@libentry()
@triton.jit
def _scatter_nonzero(
    inp,
    work,
    out,
    N,
    NB,
    shape,
    ndim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(inp + offsets, mask=offsets < N, other=0)
    selected = (offsets < N) & (values != 0)
    local = tl.cumsum(selected.to(tl.int32), axis=0) - 1
    row = (tl.load(work + NB + pid) + local).to(tl.int64)
    flat = offsets.to(tl.int64)
    for dim in tl.static_range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        coordinate = flat % dim_size
        flat = flat // dim_size
        tl.store(out + row * ndim + dim, coordinate, mask=selected)


def _can_use_block_scan(inp):
    # Metadata-only guard, independent of values/density. The scan has <=4096
    # lanes and the exact total fits int32. Other signatures retain the generic
    # FlagGems implementation; no contiguous copy is hidden in the fast path.
    return (
        inp.device.type != "cpu"
        and inp.layout == torch.strided
        and inp.dtype == torch.float32
        and inp.is_contiguous()
        and 0 < inp.numel() <= _BLOCK * _MAX_BLOCKS
    )


def nonzero(inp, *, as_tuple=False):
    """Stable row-major nonzero with exact allocation after one status read.

    The accelerated domain is contiguous FP32 with 1..2**22 elements (including
    scalars). Empty inputs need only metadata. All other nonempty signatures
    use the generic FlagGems path, whose costs are outside this optimization.
    """
    logger.debug("GEMS HYGON NONZERO")
    ndim, n = inp.ndim, inp.numel()
    if n == 0:
        out = torch.empty((0, ndim), dtype=torch.int64, device=inp.device)
    elif not _can_use_block_scan(inp):
        out = _generic_nonzero(inp, as_tuple=False)
    else:
        nb = triton.cdiv(n, _BLOCK)
        with torch_device_fn.device(inp.device):
            # NB counts followed by NB exclusive offsets; status is a separate
            # scalar to avoid a dispatching tensor index just to read the total.
            work = torch.empty((2 * nb,), dtype=torch.int32, device=inp.device)
            total = torch.empty((), dtype=torch.int32, device=inp.device)
            shape = torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)
            _count_nonzero[(nb,)](inp, work, n, BLOCK=_BLOCK, num_warps=4)
            _scan_counts[(1,)](
                work, total, nb, SCAN=triton.next_power_of_2(nb), num_warps=4
            )
            # Required device-to-host cardinality transfer / stream wait. Time
            # the full API, including this read and the following allocation.
            z = int(total.item())
            out = torch.empty((z, ndim), dtype=torch.int64, device=inp.device)
            if z and ndim:
                _scatter_nonzero[(nb,)](
                    inp,
                    work,
                    out,
                    n,
                    nb,
                    shape,
                    ndim,
                    BLOCK=_BLOCK,
                    num_warps=4,
                )
    if not as_tuple:
        return out
    if ndim == 0:
        # PyTorch treats a scalar as a length-one vector for tuple output.
        return (torch.zeros((out.shape[0],), dtype=torch.int64, device=inp.device),)
    # Metadata views only: no payload computation or second status read.
    return tuple(out.unbind(dim=1))
