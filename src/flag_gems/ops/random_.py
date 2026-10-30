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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)

_INT_TYPES = (
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
)

BLOCK_SIZE = 128


@libentry()
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "N", "from_", "to"])
def random_kernel(
    out_ptr,
    N,
    from_,
    to,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
    IS_INT64: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)

    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    c0 += i
    z = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, z, z)

    from_64 = from_.to(tl.int64)
    to_64 = to.to(tl.int64)
    # `to - from_` is computed in int64 first; for the default int64 range the
    # difference 2**63 wraps to -2**63, which reinterprets to the correct
    # unsigned span (2**63) below.
    span = (to_64 - from_64).to(tl.uint64)

    if IS_INT64:
        # Each philox counter yields two 64-bit draws (two 32-bit words each)
        # so the full int64 range (default [0, 2**63)) is reachable.
        off0 = pid * 2 * BLOCK + tl.arange(0, BLOCK)
        off1 = off0 + BLOCK
        v0 = (r0.to(tl.uint64) | (r1.to(tl.uint64) << 32)) % span
        v1 = (r2.to(tl.uint64) | (r3.to(tl.uint64) << 32)) % span
        tl.store(
            out_ptr + off0,
            (v0.to(tl.int64) + from_64).to(out_ptr.dtype.element_ty),
            mask=off0 < N,
        )
        tl.store(
            out_ptr + off1,
            (v1.to(tl.int64) + from_64).to(out_ptr.dtype.element_ty),
            mask=off1 < N,
        )
    else:
        off0 = pid * 4 * BLOCK + tl.arange(0, BLOCK)
        off1 = off0 + BLOCK
        off2 = off1 + BLOCK
        off3 = off2 + BLOCK
        v0 = (r0.to(tl.uint64) % span).to(tl.int64) + from_64
        v1 = (r1.to(tl.uint64) % span).to(tl.int64) + from_64
        v2 = (r2.to(tl.uint64) % span).to(tl.int64) + from_64
        v3 = (r3.to(tl.uint64) % span).to(tl.int64) + from_64
        tl.store(out_ptr + off0, v0.to(out_ptr.dtype.element_ty), mask=off0 < N)
        tl.store(out_ptr + off1, v1.to(out_ptr.dtype.element_ty), mask=off1 < N)
        tl.store(out_ptr + off2, v2.to(out_ptr.dtype.element_ty), mask=off2 < N)
        tl.store(out_ptr + off3, v3.to(out_ptr.dtype.element_ty), mask=off3 < N)


def _default_range(dtype):
    """Exclusive upper bound of the aten default sampling range per dtype.

    aten fills the tensor with discrete uniform samples from [0, dtype_max),
    where dtype_max is 2**bits for unsigned types and 2**(bits - 1) for signed
    ones.
    """
    if dtype == torch.bool:
        return 2
    return int(torch.iinfo(dtype).max) + 1


def random_(self, *, generator=None):
    logger.debug("GEMS RANDOM_")
    if self.dtype not in _INT_TYPES:
        raise RuntimeError(f"random_ not implemented for '{self.dtype}'")
    return random_from(self, 0, _default_range(self.dtype), generator=generator)


def random_from(self, from_, to, *, generator=None):
    logger.debug("GEMS RANDOM_ FROM")
    if self.dtype not in _INT_TYPES:
        raise RuntimeError(f"random_ not implemented for '{self.dtype}'")
    if to <= from_:
        raise RuntimeError(
            f"random_ expects 'from' to be less than 'to', but got from={from_} to={to}"
        )
    N = volume(self.shape)
    is_int64 = self.dtype == torch.int64
    # int64 needs two 32-bit words per draw; the other dtypes reuse the four
    # words produced by a single philox counter.
    UNROLL = 2 if is_int64 else 4
    grid_fn = lambda meta: (triton.cdiv(N, meta["BLOCK"] * UNROLL),)
    increment = triton.cdiv(N, UNROLL)
    philox_seed, philox_offset = philox_backend_seed_offset(
        increment, generator=generator
    )
    with torch_device_fn.device(self.device):
        random_kernel[grid_fn](
            self, N, from_, to, philox_seed, philox_offset, BLOCK_SIZE, is_int64
        )
    return self
