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

"""Triton implementation of aten::_foreach_addcdiv_.Scalar (in-place).

For every list index i:
    self_i = self_i + value * (tensor1_i / tensor2_i)     (fp32 opmath, one final rounding)

Two launch plans, chosen per call from the list's total element count:
  * small lists -> one grouped launch covering up to 16 triples (there the launch
    count is the dominant cost);
  * large lists -> one compact kernel launch per triple, so the kernel carries no
    per-program slot-selection prologue and CPU enqueue time hides behind the
    bandwidth-bound GPU work.
"""

import torch
import triton
import triton.language as tl

_GROUP = 16
# Below this many elements (summed over the list) a call is launch-bound and is
# served by a single grouped launch; above it the kernel is bandwidth-bound and
# per-triple launches are cheaper on the device.
_SMALL_ELEMS = 1 << 20


@triton.jit
def _addcdiv_body(s_ptr, a_ptr, b_ptr, alpha, pid_local, n):
    offs = pid_local * 1024 + tl.arange(0, 1024)
    mask = offs < n
    x = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = s + (alpha * x) / y
    tl.store(s_ptr + offs, out.to(s_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _addcdiv_k1(s_ptr, a_ptr, b_ptr, n, alpha):
    _addcdiv_body(s_ptr, a_ptr, b_ptr, alpha, tl.program_id(0), n)


@triton.jit
def _addcdiv_k16(
    s0,
    a0,
    b0,
    s1,
    a1,
    b1,
    s2,
    a2,
    b2,
    s3,
    a3,
    b3,
    s4,
    a4,
    b4,
    s5,
    a5,
    b5,
    s6,
    a6,
    b6,
    s7,
    a7,
    b7,
    s8,
    a8,
    b8,
    s9,
    a9,
    b9,
    s10,
    a10,
    b10,
    s11,
    a11,
    b11,
    s12,
    a12,
    b12,
    s13,
    a13,
    b13,
    s14,
    a14,
    b14,
    s15,
    a15,
    b15,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    n6,
    n7,
    n8,
    n9,
    n10,
    n11,
    n12,
    n13,
    n14,
    n15,
    alpha,
):
    pid = tl.program_id(0)
    c0 = tl.cdiv(n0, 1024)
    c1 = c0 + tl.cdiv(n1, 1024)
    c2 = c1 + tl.cdiv(n2, 1024)
    c3 = c2 + tl.cdiv(n3, 1024)
    c4 = c3 + tl.cdiv(n4, 1024)
    c5 = c4 + tl.cdiv(n5, 1024)
    c6 = c5 + tl.cdiv(n6, 1024)
    c7 = c6 + tl.cdiv(n7, 1024)
    c8 = c7 + tl.cdiv(n8, 1024)
    c9 = c8 + tl.cdiv(n9, 1024)
    c10 = c9 + tl.cdiv(n10, 1024)
    c11 = c10 + tl.cdiv(n11, 1024)
    c12 = c11 + tl.cdiv(n12, 1024)
    c13 = c12 + tl.cdiv(n13, 1024)
    c14 = c13 + tl.cdiv(n14, 1024)
    c15 = c14 + tl.cdiv(n15, 1024)
    if pid < c0:
        _addcdiv_body(s0, a0, b0, alpha, pid, n0)
    elif pid < c1:
        _addcdiv_body(s1, a1, b1, alpha, pid - c0, n1)
    elif pid < c2:
        _addcdiv_body(s2, a2, b2, alpha, pid - c1, n2)
    elif pid < c3:
        _addcdiv_body(s3, a3, b3, alpha, pid - c2, n3)
    elif pid < c4:
        _addcdiv_body(s4, a4, b4, alpha, pid - c3, n4)
    elif pid < c5:
        _addcdiv_body(s5, a5, b5, alpha, pid - c4, n5)
    elif pid < c6:
        _addcdiv_body(s6, a6, b6, alpha, pid - c5, n6)
    elif pid < c7:
        _addcdiv_body(s7, a7, b7, alpha, pid - c6, n7)
    elif pid < c8:
        _addcdiv_body(s8, a8, b8, alpha, pid - c7, n8)
    elif pid < c9:
        _addcdiv_body(s9, a9, b9, alpha, pid - c8, n9)
    elif pid < c10:
        _addcdiv_body(s10, a10, b10, alpha, pid - c9, n10)
    elif pid < c11:
        _addcdiv_body(s11, a11, b11, alpha, pid - c10, n11)
    elif pid < c12:
        _addcdiv_body(s12, a12, b12, alpha, pid - c11, n12)
    elif pid < c13:
        _addcdiv_body(s13, a13, b13, alpha, pid - c12, n13)
    elif pid < c14:
        _addcdiv_body(s14, a14, b14, alpha, pid - c13, n14)
    elif pid < c15:
        _addcdiv_body(s15, a15, b15, alpha, pid - c14, n15)


def _launch_one(s, a, b, numel, alpha):
    _addcdiv_k1[((numel + 1023) // 1024,)](s, a, b, numel, alpha)


def _launch_group(S, A, B, ns, alpha):
    total = 0
    for q in ns:
        total += (q + 1023) // 1024
    flat = []
    for k in range(len(S)):
        flat.append(S[k])
        flat.append(A[k])
        flat.append(B[k])
    pad = _GROUP - len(S)
    if pad > 0:
        flat.extend([flat[-1]] * (3 * pad))
        ns = ns + [0] * pad
    _addcdiv_k16[(total,)](*flat, *ns, alpha)


def _run_scalar(S, A, B, alpha):
    n = len(S)
    ns = [t.numel() for t in S]
    total = 0
    for q in ns:
        total += q
    if n <= _GROUP and total <= _SMALL_ELEMS:
        _launch_group(S, A, B, ns, alpha)
        return None
    for k in range(n):
        if ns[k]:
            _launch_one(S[k], A[k], B[k], ns[k], alpha)
    return None


def foreach_addcdiv_scalar_(self, tensor1, tensor2, value=1):
    n = len(self)
    if len(tensor1) != n or len(tensor2) != n:
        raise RuntimeError("_foreach_addcdiv_: tensor lists must have the same length")
    if not n:
        return None

    t = type(value)
    if t is float or t is int:
        return _run_scalar(self, tensor1, tensor2, float(value))

    if isinstance(value, torch.Tensor):
        return _run_scalar(self, tensor1, tensor2, float(value.item()))

    # ScalarList overload: one alpha per tensor triple, in order.
    for s, a, b, v in zip(self, tensor1, tensor2, value):
        numel = s.numel()
        if numel:
            _launch_one(s, a, b, numel, float(v))
    return None
