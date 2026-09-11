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

import hashlib
import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language import core as tlcore
from triton.runtime.cache import get_cache_manager

from flag_gems.ops._embedding_bag import _embedding_bag_impl as _generic_impl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


_COREX_ASSERT_SYMBOLS = tl.constexpr({(tl.int32,): ("flaggems_corex_assert", tl.int32)})
# extern_elementwise requires a result, whereas the SDK routine returns void.
# This adapter also supplies constant strings in the SDK's generic address space.
_COREX_ASSERT_IR = r"""
target triple = "bi-iluvatar-ilurt"

@embedding_bag_assert_message = private unnamed_addr addrspace(4)
    constant [39 x i8] c"embedding_bag: invalid index or offset\00", align 1
@embedding_bag_assert_file = private unnamed_addr addrspace(4) constant [18 x i8] c"_embedding_bag.py\00", align 1
@embedding_bag_assert_function = private unnamed_addr addrspace(4) constant [15 x i8] c"_embedding_bag\00", align 1

declare void @__assertfail(ptr, ptr, i32, ptr, i64)

define i32 @flaggems_corex_assert(i32 %code) alwaysinline nounwind {
entry:
  %ok = icmp eq i32 %code, 0
  br i1 %ok, label %end, label %invalid
invalid:
  call void @__assertfail(
      ptr addrspacecast (ptr addrspace(4) @embedding_bag_assert_message to ptr),
      ptr addrspacecast (ptr addrspace(4) @embedding_bag_assert_file to ptr),
      i32 1,
      ptr addrspacecast (ptr addrspace(4) @embedding_bag_assert_function to ptr),
      i64 1)
  br label %end
end:
  ret i32 %code
}
"""


@lru_cache(maxsize=1)
def _corex_assert_library():
    # FileCacheManager publishes with an atomic rename. Identical concurrent
    # writers are safe; changing the contents must produce a different path
    # because CoreX memoizes external-library hashes by path.
    key = hashlib.sha256(
        (
            "flag_gems.embedding_bag.assert.v1\0"
            + triton.__version__
            + _COREX_ASSERT_IR
        ).encode("utf-8")
    ).hexdigest()
    cache = get_cache_manager(key)
    name = "embedding_bag_assert.ll"
    path = cache.get_file(name)
    if path is None:
        path = cache.put(_COREX_ASSERT_IR, name, binary=False)
    # Triton's final kernel key additionally includes the SDK libdevice hash.
    return path


@triton.jit
def _embedding_bag_corex_assert(code):
    # CoreX 4.4 lowers tl.device_assert to printf only. Call the SDK's actual
    # assertion routine through Triton's existing external-library linker.
    empty: tl.constexpr = ""
    pure: tl.constexpr = False
    tlcore.extern_elementwise(
        empty.value,
        empty.value,
        [code],
        _COREX_ASSERT_SYMBOLS.value,
        is_pure=pure.value,
    )


@libentry()
@triton.jit(
    do_not_specialize=[
        "error",
        "acc_bits",
        "out_bits",
        "freq",
        "num_error_flags",
        "num_acc_elements",
        "embedding_dim",
        "kind",
        "count_freq",
    ]
)
def _corex_embedding_bag_check_flags(
    error,
    acc_bits,  # accumulator pointer bits
    out_bits,  # output pointer bits
    freq,  # embedding occurrence frequencies
    num_error_flags,
    num_acc_elements,  # number of accumulator elements
    embedding_dim,
    kind,
    count_freq,  # count or normalize by embedding frequency
    block: tl.constexpr,  # element block size
):
    # Fixed pointer and integer signatures keep every caller in one SDK module.
    # The top bit makes all extents uint64 scalars, including small/empty calls.
    n = (num_error_flags & 0x7FFFFFFFFFFFFFFF).to(tl.int64)
    a = (num_acc_elements & 0x7FFFFFFFFFFFFFFF).to(tl.int64)
    d = (embedding_dim & 0x7FFFFFFFFFFFFFFF).to(tl.int64)
    lane = tl.arange(0, block)
    x = tl.program_id(0).to(tl.int64) * block + lane
    if a > 0:
        acc = acc_bits.to(tl.pointer_type(tl.float32))
        values = tl.load(acc + x, x < a, other=0)
        if count_freq != 0:
            frequency = tl.load(freq + x // d, x < a, other=1)
            values = values / tl.maximum(frequency, 1).to(tl.float32)
        if kind == 1:
            target16 = out_bits.to(tl.pointer_type(tl.float16))
            tl.store(target16 + x, values.to(tl.float16), x < a)
        elif kind == 2:
            target_bf16 = out_bits.to(tl.pointer_type(tl.bfloat16))
            tl.store(target_bf16 + x, values.to(tl.bfloat16), x < a)
        else:
            target32 = out_bits.to(tl.pointer_type(tl.float32))
            tl.store(target32 + x, values, x < a)
    if tl.program_id(0) == 0:
        bad = tl.full((), 0, tl.int32)
        for base in range(0, n, block):
            code = tl.load(error + base + lane, base + lane < n, other=0)
            bad |= tl.max(code, 0)
        _embedding_bag_corex_assert(bad)


def _check_corex_error(error, count, finish=None):
    if finish is None:
        acc = out = frequencies = error
        a, d, kind, scale = 0, 1, 0, 0
    else:
        pointers, metadata = finish
        acc, frequencies, out = pointers
        a, d, scale, _ = metadata
        kind = {torch.float16: 1, torch.bfloat16: 2, torch.float32: 3}[out.dtype]
    with torch_device_fn.device(error.device):
        _corex_embedding_bag_check_flags[(max(triton.cdiv(a, 4096), 1),)](
            triton.reinterpret(error, tl.int32),
            triton.reinterpret(acc, tl.int32),
            triton.reinterpret(out, tl.int32),
            triton.reinterpret(frequencies, tl.int32),
            count + (1 << 63),
            a + (1 << 63),
            d + (1 << 63),
            int(kind),
            int(scale),
            4096,
            debug=False,
            extern_libs={"embedding_bag_assert": _corex_assert_library()},
        )


def _embedding_bag(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    logger.debug("GEMS_ILUVATAR _EMBEDDING_BAG")
    return _generic_impl(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
        _async_assert=False,
        _check_error=_check_corex_error,
    )


def _embedding_bag_forward_only(
    weight,
    indices,
    offsets,
    scale_grad_by_freq=False,
    mode=0,
    sparse=False,
    per_sample_weights=None,
    include_last_offset=False,
    padding_idx=-1,
):
    logger.debug("GEMS_ILUVATAR _EMBEDDING_BAG_FORWARD_ONLY")
    return _generic_impl(
        weight,
        indices,
        offsets,
        scale_grad_by_freq,
        mode,
        sparse,
        per_sample_weights,
        include_last_offset,
        padding_idx,
        forward_only=True,
        _async_assert=False,
        _check_error=_check_corex_error,
    )
