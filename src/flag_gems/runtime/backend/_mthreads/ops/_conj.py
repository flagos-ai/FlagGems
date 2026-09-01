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
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("_conj"),
    key=["n_slots"],
)
@triton.jit
def _conj_kernel(in_ptr, out_ptr, n_slots, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    # Use int64 offsets since n_slots = 2 * numel can reach 2**31 for large
    # tensors (e.g. 1G complex64 elements) and overflow int32.
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < n_slots

    # Complex numbers are stored as interleaved (real, imag) float slots.
    # A flat coalesced load/store over all slots keeps 128-bit vectorization;
    # the strided two-slot access (base = offs * 2) used by the generic kernel
    # only generates 32-bit scalar loads/stores on MUSA (~2.5-3x slower).
    v = tl.load(in_ptr + offs, mask=mask)
    v = tl.where((offs & 1) == 1, -v, v)  # negate imaginary slots
    tl.store(out_ptr + offs, v, mask=mask)


def _conj_physical_copy(input: torch.Tensor) -> torch.Tensor:
    """Materialize the conjugate with a vectorized copy kernel."""
    n_slots = 2 * input.numel()
    src = input if input.is_contiguous() else input.contiguous()
    output = torch.empty_like(src)
    in_ptr = torch.view_as_real(src)
    out_ptr = torch.view_as_real(output)

    grid = lambda meta: (triton.cdiv(n_slots, meta["BLOCK_SIZE"]),)
    _conj_kernel[grid](in_ptr, out_ptr, n_slots)
    return output


def _conj(input: torch.Tensor) -> torch.Tensor:
    """Complex conjugate of ``input`` (mthreads/MUSA specialization).

    ``torch._conj`` (aten::_conj) is a *pure view* operation in eager PyTorch:
    it allocates a lightweight alias sharing the same storage and merely
    toggles the tensor's conjugate bit (~0.00036 ms, zero data traffic).
    Materializing it with a Triton kernel (as the generic FlagGems
    implementation does) is both a semantic deviation from eager semantics and
    a massive latency regression for large tensors. For complex64/complex128
    we reproduce the zero-copy conjugate-bit view exactly (same approach as
    the _thead backend).

    torch_musa cannot materialize complex32 (ComplexHalf) conjugate-bit views
    ("SetMUTensorDType Unsupported tensor dtype: ComplexHalf"), so complex32
    falls back to a physical copy with the vectorized kernel above.
    """
    logger.debug("GEMS CONJ")

    if not input.is_complex():
        # Real tensors: conjugate is the identity; mirror torch._conj's
        # zero-copy alias behavior.
        return input.as_strided(input.size(), input.stride(), input.storage_offset())

    if input.is_conj():
        # conj(conj(x)) = x: toggle the conjugate bit off, still zero-copy.
        out = input.as_strided(input.size(), input.stride(), input.storage_offset())
        torch._C._set_conj(out, False)
        return out

    if input.dtype == torch.complex32:
        # torch_musa cannot read back conj-bit complex32 views correctly.
        return _conj_physical_copy(input)

    # complex64/complex128: build a storage-sharing alias with the conjugate
    # bit set, matching eager ``torch._conj`` semantics (involution included).
    out = input.as_strided(input.size(), input.stride(), input.storage_offset())
    torch._C._set_conj(out, True)
    return out
