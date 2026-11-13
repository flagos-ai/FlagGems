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
"""GCU400 argsort dispatch.

Inside the mradix envelope (up to MAX_SORT_COLS elements along the sorted axis,
a dtype the key builder covers) this is the dedicated TLE-Struct + TLE-Raw
arg-sort in argsort_mradix.py / argsort_mradix_l1.tops. Everything else --
int64 and bool rows, rows longer than the VDMEM envelope -- falls back to
``sort(...)[1]``, which is the composition torch uses and is the working generic
radix on this backend.

Note that the generic ``flag_gems.ops.argsort`` cannot be the fallback here: its
merge-sort kernels do not compile for GCU (they trip an arith.select type
mismatch), which is why argsort had no working implementation on this backend
before the mradix path existed.
"""

import logging

import torch

from . import argsort_mradix as _argsort_mradix
from .sort import sort

logger = logging.getLogger(__name__)


def argsort(inp, dim=-1, descending=False, stable=False):
    logger.debug("GEMS_ENFLAME ARGSORT")

    rank = max(inp.ndim, 1)
    if dim < -rank or dim >= rank:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-rank}, "
            f"{rank - 1}], but got {dim})"
        )
    if inp.ndim == 0:
        # A scalar has exactly one position; its only valid dims are 0 and -1.
        return torch.zeros((), dtype=torch.int64, device=inp.device)
    if inp.numel() == 0:
        return torch.empty(inp.shape, dtype=torch.int64, device=inp.device)

    n = inp.shape[dim]
    if n == 1:
        # A one-element axis ranks to 0 everywhere; skip the digit pipeline and
        # the position store entirely.
        return torch.zeros(inp.shape, dtype=torch.int64, device=inp.device)

    # MRADIX path. The kernel ranks a contiguous (rows, n) view of the sorted
    # axis, so a non-last `dim` is moved last and the result moved back; that
    # keeps it on the same kernel instead of the generic fallback. The mradix
    # sort is a stable LSD radix by construction, which is exactly the tie-break
    # `stable=True` asks for; `stable=False` may return any tie order, so the
    # same kernel serves both.
    _ = stable
    if (
        _argsort_mradix.argsort_mradix_dtype_ok(inp.dtype)
        and n <= _argsort_mradix.MAX_SORT_COLS
    ):
        moved_axis = dim != -1 and dim != rank - 1
        if moved_axis:
            # A non-last `dim` is moved last and the result moved back, so it
            # still runs the mradix kernel instead of the generic fallback
            # (which is both slower and does not match torch's comparison order
            # for +-0.0 / NaN, see _to_sort_key).
            moved = inp.movedim(dim, -1)
            out_shape = moved.shape
            flat = moved.contiguous().reshape(-1, n)
        else:
            # The common case stays on metadata-only ops: a contiguous input is
            # returned as is by contiguous() and reshaped as a view, which
            # matters because a small row set only takes a few microseconds.
            out_shape = inp.shape
            flat = inp.contiguous().reshape(-1, n)
        idx = _argsort_mradix.argsort_mradix(flat, descending)
        # The kernel returns int64 positions itself when the row fills the tile;
        # a partial row still comes back as int32 (a masked 64-bit store loses
        # lanes on GCU) and is widened here.
        if idx.dtype != torch.int64:
            idx = idx.to(torch.int64)
        idx = idx.reshape(out_shape)
        if moved_axis:
            idx = idx.movedim(-1, dim).contiguous()
        return idx

    # Generic radix. sort() owns the dtypes the mradix envelope does not cover
    # (int64, bool) and the rows that do not fit the VDMEM windows.
    return sort(inp, dim=dim, descending=descending)[1]
