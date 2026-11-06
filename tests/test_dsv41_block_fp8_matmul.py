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

from flag_gems.ops.w8a8_block_fp8_matmul import w8a8_block_fp8_matmul
from tests.sm90_fp8_helpers import (
    assert_exact,
    changed_graph,
    fp8_inputs,
    fp8_reference,
    native_block_fp8_baseline,
)


def _matmul(A, B, As, Bs, output_dtype=torch.bfloat16):
    return w8a8_block_fp8_matmul(A, B, As, Bs, [32, 32], output_dtype)


pytestmark = [
    pytest.mark.dsv41_block_fp8_matmul,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
        reason="SM90 operator",
    ),
]

ROWS = [
    3,
    5,
    6,
    9,
    10,
    12,
    15,
    18,
    20,
    21,
    24,
    25,
    30,
    35,
    36,
    40,
    42,
    48,
    50,
    60,
    72,
    80,
    84,
    96,
    100,
    120,
]


@pytest.mark.parametrize("m", ROWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_actual_decode_row_buckets(m, dtype):
    # Attention projection plus fallback-only boundary rows use the same K32 contract.
    args = fp8_inputs(m, 1792, 5120)
    fn = lambda *x: _matmul(*x, output_dtype=dtype)
    baseline = lambda *x: native_block_fp8_baseline(*x, output_dtype=dtype)
    assert_exact(fn(*args), baseline(*args))
    expected = fp8_reference(*args, output_dtype=dtype)
    # FP8 MMA's native accumulation already differs from FP32 Torch. The
    # established port reference uses rtol=.02/atol=.25; this tighter semantic
    # check is separate from the unchanged zero-error native regression gate.
    torch.testing.assert_close(
        fn(*args).float(), expected.float(), rtol=0.01, atol=0.01
    )

    def refresh(xs, seed):
        fresh = fp8_inputs(m, 1792, 5120, 100 + seed)
        for x, y in zip(xs, fresh):
            x.copy_(y)

    changed_graph(fn, args, refresh, baseline)


@pytest.mark.parametrize(
    "n,k",
    [
        (16384, 1280),
        (5120, 4096),
        (4608, 5120),
        (5120, 2304),
        (4096, 1280),
        (25600, 6144),
    ],
)
@pytest.mark.parametrize("m", [24, 48, 96])
def test_full_projection_families(m, n, k):
    args = fp8_inputs(m, n, k)
    a = _matmul(*args)
    b = native_block_fp8_baseline(*args)
    assert_exact(a, b)
    for _ in range(10):
        assert_exact(_matmul(*args), a)


def test_tail_empty_and_scale_guards():
    args = fp8_inputs(7, 35, 128)
    torch.testing.assert_close(
        _matmul(*args),
        fp8_reference(*args),
        rtol=0.01,
        atol=0.01,
    )
    args = fp8_inputs(0, 35, 128)
    assert _matmul(*args).shape == (0, 35)
    a, b, sa, sb = fp8_inputs(1, 32, 32)
    with pytest.raises(TypeError, match="numeric scales"):
        _matmul(a, b, sa.to(torch.int32), sb)
