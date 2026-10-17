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
from flag_gems.fused.dsv41.fp4_indexer import fp4_index_logits_decode
from tests.dsv41_helpers import assert_exact, changed_graph, fp4_inputs, fp4_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_direct_slot_decode_matches_independent_reference():
    args = fp4_inputs(6, 129)
    q, weights, mapping, req, lens, table, page, _, width = args
    slots = mapping[req, :width].to(torch.int64).contiguous()
    actual = fp4_index_logits_decode(q, weights, slots, lens, table, page)
    torch.testing.assert_close(actual, fp4_reference(*args), rtol=0.008, atol=0.03125)


@pytest.mark.parametrize("width", [0, 1, 63, 64, 65, 129])
@pytest.mark.parametrize("page", [64, 256])
def test_explicit_slots_reference_and_changed_graph(width, page):
    args = fp4_inputs(7, width, page=page)
    q, w, mapping, req, lens, table, _, _, _ = args
    slots = mapping[req, :width].long().contiguous()
    xs = (q, w, slots, lens, table)
    fn = lambda *x: fp4_index_logits_decode(*x, page)
    if width:
        expected = fp4_reference(*args)
        torch.testing.assert_close(fn(*xs), expected, rtol=0.008, atol=0.03125)

        def refresh(values, seed):
            values[0].copy_(torch.roll(values[0], seed + 1, 1))
            values[1].mul_(0.9375)
            values[2].copy_(torch.roll(values[2], seed + 1, 1))
            values[3].copy_(
                torch.arange(7, device="cuda")
                .remainder(4)
                .neg()
                .add(width)
                .clamp_min(0)
            )

        # Compare capture/replay against a fresh non-captured call with updated
        # physical slots and visible lengths on every replay.
        changed_graph(fn, xs, refresh, fn)
    else:
        assert fn(*xs).shape == (7, 0)
