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

"""Exact E2M1 storage words to four E4M3 bytes, independent of register layout.

Bits 0..15 hold four FP4 nibbles in increasing coordinate order. The result
holds four FP8 bytes in the same order. Both FP4 zero signs produce +0,
matching the frozen BF16 decode contract. No scale is applied here.
"""

import triton
import triton.language as tl


@triton.jit
def decode_fp4_e2m1_words_to_fp8x4(words):
    return tl.inline_asm_elementwise(
        """{
        .reg .b32 idx,mag,sgn,nz,tmp;
        and.b32 idx,$1,0x7777;
        prmt.b32 mag,0x3c383000,0x4c484440,idx;
        and.b32 idx,$1,0x8888;
        shr.u32 idx,idx,1;
        prmt.b32 sgn,0,0x80808080,idx;
        shr.u32 nz,mag,4;
        shr.u32 tmp,mag,6;
        or.b32 nz,nz,tmp;
        and.b32 nz,nz,0x01010101;
        shl.b32 nz,nz,7;
        and.b32 sgn,sgn,nz;
        or.b32 $0,mag,sgn;
        }""",
        constraints="=r,r",
        args=[words],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
