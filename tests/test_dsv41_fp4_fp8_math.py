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

import pytest
import torch
import triton

pytest.importorskip("triton.experimental.gluon")
from triton.experimental import gluon as g  # noqa: E402
from triton.experimental.gluon import language as gl  # noqa: E402

from flag_gems.fused.dsv41.fp4_fp8 import (  # noqa: E402
    e2m1x4_to_fp8,
    normal_ue8m0_scale,
    prepare_fp4_query,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (9, 0)
    or triton.__version__.split("+")[0] not in ("3.6.0", "3.7.1"),
    reason="validated SM90 / Triton 3.6.0 / 3.7.1 path",
)


@g.jit
def _decode_words(words, output, n: gl.constexpr):
    x = gl.program_id(0) * 512 + gl.arange(
        0, 512, layout=gl.BlockedLayout([4], [32], [4], [0])
    )
    packed = gl.load(words + x // 4, x < n, other=0).to(gl.uint32)
    gl.store(output + x, e2m1x4_to_fp8(packed), x < n)


def test_every_packed_word_including_negative_zero():
    words = torch.arange(65536, device="cuda", dtype=torch.int32)
    out = torch.empty(65536 * 4, device="cuda", dtype=torch.float8_e4m3fn)
    _decode_words[(triton.cdiv(out.numel(), 512),)](words, out, out.numel())
    codes = (words[:, None] >> (4 * torch.arange(4, device="cuda"))[None, :]) & 15
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    ref = lut[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    ref = torch.where((codes & 7) == 0, 0.0, ref).to(torch.float8_e4m3fn).flatten()
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8))


def test_query_restore_and_reject_nonrepresentable_values():
    q = torch.full((6, 32, 128), 6.0, device="cuda", dtype=torch.bfloat16)
    q[:, :, 1::2] = -1.5
    q8 = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    scales = torch.empty((6, 32, 4), device="cuda")
    ok = torch.empty((6, 32), device="cuda", dtype=torch.int32)
    prepare_fp4_query[(6,)](q, q8, scales, ok, num_warps=4)
    assert bool((ok == 1).all())
    restored = (q8.float().reshape(6, 32, 4, 32) * scales[..., None]).reshape_as(q)
    assert torch.equal(restored, q.float())
    q[0, 0, 0] = 1.1015625
    q[0, 1, :32] *= 64
    prepare_fp4_query[(6,)](q, q8, scales, ok, num_warps=4)
    assert ok[0, 0].item() == 0 and ok[0, 1].item() == 0
    assert bool((ok[1:] == 1).all())


@g.jit
def _normal_scales(output):
    encoded = gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [4], [0]))
    gl.store(
        output + encoded, normal_ue8m0_scale(encoded), (encoded > 0) & (encoded < 255)
    )


def test_every_normal_ue8m0_scale_is_exact():
    output = torch.empty(256, dtype=torch.float32, device="cuda")
    _normal_scales[(1,)](output)
    encoded = torch.arange(1, 255, dtype=torch.int32, device="cuda")
    expected = torch.exp2((encoded - 127).float())
    assert torch.equal(output[1:255].view(torch.uint8), expected.view(torch.uint8))
