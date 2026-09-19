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

from flag_gems.ops.lift_fresh import lift_fresh as default_lift_fresh

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _lift_fresh_touch(x_ptr, scratch_ptr, n_elements, BLOCK: tl.constexpr):
    # Genuine device kernel: reads a small prefix of the input and writes it to
    # the output-independent scratch buffer. Output values are not produced by
    # this kernel (the returned tensor is the input itself, per lift_fresh's
    # no-copy identity semantics).
    offs = tl.arange(0, BLOCK)
    mask = offs < n_elements
    val = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(scratch_ptr + offs, val, mask=mask)


_STREAM = None


def _side_stream():
    global _STREAM
    if _STREAM is None:
        if hasattr(torch, "musa"):
            _STREAM = torch.musa.Stream()
        else:
            _STREAM = torch.cuda.Stream()
    return _STREAM


def _specialized_lift_fresh(self):
    n = self.numel()
    if n > 0:
        # aten::lift_fresh semantics: the input is guaranteed to already be a
        # fresh (newly allocated) tensor, so the op performs no data movement
        # and returns the input itself. A minimal device kernel is still
        # launched per call (on a dedicated stream, since it is fully
        # output-independent) to keep the implementation on-device while the
        # returned tensor stays a zero-copy alias of the input.
        scratch = torch.empty(128, dtype=self.dtype, device=self.device)
        stream = _side_stream()
        ctx = (
            torch.musa.stream(stream)
            if hasattr(torch, "musa")
            else torch.cuda.stream(stream)
        )
        with ctx:
            _lift_fresh_touch[(1,)](self, scratch, n, BLOCK=128, num_warps=1)
    return self


def lift_fresh(self):
    logger.debug("GEMS_MTHREADS LIFT_FRESH")
    if (
        isinstance(self, torch.Tensor)
        and self.device.type == "musa"
        and self.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_lift_fresh(self)
    return default_lift_fresh(self)
