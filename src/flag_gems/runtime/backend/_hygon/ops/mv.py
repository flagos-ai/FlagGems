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

from .mm import mv as _mv


# Pointer specialization is disabled so the kernel accepts contiguous views
# with different alignment or backing-storage sizes.
@triton.jit(
    do_not_specialize=["A", "B", "C"],
    do_not_specialize_on_alignment=["A", "B", "C"],
)
def _mv_small_kernel(A, B, C, N: tl.constexpr, M: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, M)
    a = tl.load(A + row * M + k).to(tl.float32)
    b = tl.load(B + k).to(tl.float32)
    value = tl.sum(a * b, axis=0)
    tl.store(C + row, value)


def _launch_small_mv(inp, vec, out, n, m):
    if inp.device.index != torch_device_fn.current_device():
        with torch_device_fn.device(inp.device):
            _launch_small_mv(inp, vec, out, n, m)
        return
    _mv_small_kernel[(n,)](
        inp,
        vec,
        out,
        n,
        m,
        num_warps=2 if m == 2048 else 4,
        num_stages=2,
    )


def mv(inp, vec):
    return _mv(inp, vec)
