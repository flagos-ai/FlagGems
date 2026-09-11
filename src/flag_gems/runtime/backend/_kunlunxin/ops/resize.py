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
import logging

import torch

logger = logging.getLogger(__name__)


def resize(inp: torch.Tensor, size, memory_format=None):
    """Out-of-place resize (kunlunxin / XPU).

    The generic implementation copies the data with a Triton kernel
    (``_resize_kernel``). On XPU that contiguous copy is ~22x slower than the
    vendor's native copy engine (e.g. [4096, 4096] fp16: 0.88ms vs 0.04ms),
    which makes every ``resize`` call pure overhead (native resize copies the
    preserved elements with the DMA engine and is essentially free).

    Fix: allocate the output and copy the preserved ``min(old, new)`` elements
    through the ATen ``_copy_from`` primitive. gems overrides ``copy_``/``copy``
    but never ``_copy_from``, so this reaches the native strided-copy engine and
    runs at native speed even while use_gems is active. Result matches native
    ``aten.resize`` exactly (output is a fresh tensor, not an alias).
    """
    logger.debug("GEMS RESIZE")

    if not isinstance(size, tuple):
        size = tuple(size)

    out = torch.empty(size, device=inp.device, dtype=inp.dtype)

    if inp.numel() == 0 or out.numel() == 0:
        return out

    # resize preserves the first min(old_numel, new_numel) elements; the rest
    # (when growing) is left uninitialized, matching native semantics.
    copy_numel = min(inp.numel(), out.numel())
    src = inp.reshape(-1)[:copy_numel]
    dst = out.reshape(-1)[:copy_numel]
    # Native contiguous copy (bypasses the slow gems/Triton copy path).
    torch.ops.aten._copy_from(src, dst, False)

    return out


def resize_(inp: torch.Tensor, size, memory_format=None):
    logger.debug("GEMS RESIZE_")

    if not isinstance(size, tuple):
        size = tuple(size)

    inp.set_(inp.untyped_storage(), 0, size)
    return inp
