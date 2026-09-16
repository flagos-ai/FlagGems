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

logger = logging.getLogger(__name__)


def empty(
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
):
    """Returns a tensor filled with uninitialized data (kunlunxin XPU).

    The generic flag_gems.ops.empty launches an additional Triton zero-fill
    kernel (``empty_kernel`` with a masked ``tl.store(ptr, 0.0)``).  On XPU
    that kernel costs a multi-second one-off JIT compile plus a millisecond
    scale warm-up, making small-shape ``gems empty`` ~70ms.  ``empty`` has
    uninitialized semantics, so this specialization skips the zero-fill
    kernel entirely and returns the raw ``torch.empty_strided`` allocation.
    No in-repo caller of ``gems empty`` relies on the zero value: the
    kunlunxin fused users (sparse_mla, mhc_bwd, bf16_paged_mqa_logits,
    hc_split_sinkhorn) treat the result purely as an output buffer that
    their kernels fully overwrite (mhc_bwd only calls it for an empty shape).
    """
    logger.debug("GEMS_KUNLUNXIN EMPTY")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        import flag_gems.runtime as _rt

        device = torch.device(_rt.device.name)
    if layout is None:
        layout = torch.strided
    if pin_memory is None:
        pin_memory = False
    if memory_format is None:
        memory_format = torch.contiguous_format

    # Mirror the generic impl: meta gives the strides for the requested
    # memory_format, empty_strided does the real allocation.  This avoids
    # self-recursion through aten::empty.memory_format and is safe on XPU
    # (aten::empty is not intercepted on this branch).
    shape = tuple(size)
    meta = torch.empty(shape, dtype=dtype, device="meta", memory_format=memory_format)
    return torch.empty_strided(
        shape,
        meta.stride(),
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
    )
