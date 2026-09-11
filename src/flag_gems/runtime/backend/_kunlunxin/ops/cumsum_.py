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

from .cumsum import cumsum_wrapper

logger = logging.getLogger(__name__)


def cumsum_(inp, dim=1, *, dtype=None):
    """In-place prefix sum on the Kunlunxin backend.

    The generic implementation routes through the *generic* ``cumsum_wrapper``
    plus a trailing ``inp.copy_(result)``. On XPU that is both wrong and slow:

    * wrong -- the generic K > 1 (mid-dim) scan misses the
      ``TRITONXPU_OTHER_SIM`` / ``TRITONXPU_STORE_MASK_SIM`` guards that the
      vendor scan uses, so the masked tail block reads adjacent memory and the
      whole (200, 40999, 3) family mis-computes for every dtype;
    * slow -- an extra ``empty_like`` (int inputs even widen to int64) plus a
      full extra device-to-device copy on top of the scan.

    So this override reuses the vendor scan (row / chunked / identity tiers of
    ``_kunlunxin.ops.cumsum``) and lets it write straight into ``inp``. The
    aliasing is safe for every tier because each program only ever stores to
    addresses it has already loaded in the same iteration (load-before-store),
    and never re-loads an address it wrote, so no ``tl.debug_barrier()`` is
    required between the read and the write side.
    """
    logger.debug("GEMS_KUNLUNXIN CUMSUM_")
    if dtype is not None and dtype != inp.dtype:
        raise RuntimeError(
            "Bad in-place call: input tensor dtype and output tensor dtype should match"
        )
    assert inp.dtype in (
        torch.float16,
        torch.float32,
        torch.bfloat16,
        torch.int16,
        torch.int32,
    ), f"cumsum_ only supports float/int dtypes, got {inp.dtype}"

    if inp.numel() == 0:
        return inp

    if inp.is_contiguous():
        # `cumsum_wrapper` calls `.contiguous()` internally, which is a no-op
        # here, so the scan writes the result directly back into `inp`.
        cumsum_wrapper(inp, dim, inp.dtype, out=inp)
    else:
        # Non-contiguous self: scan the contiguous copy, then write back with
        # the native strided-copy engine (`_copy_from` is not overridden by
        # flag_gems, so this avoids recursing into the gems `copy_`).
        result = cumsum_wrapper(inp, dim, inp.dtype)
        torch.ops.aten._copy_from(result, inp, False)
    return inp
