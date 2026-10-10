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

"""Public-API regression tests for the split-KV combine path (#6872).

`mha_fwd` selects split-KV on its own; only the device SM count is overridden so
that the choice is deterministic. The real split and combine kernels still run.
"""

import pytest
import torch
import triton

import flag_gems
from flag_gems.ops import flash_api

from .test_flash_kernel import DTYPES, LAYOUTS, assert_no_stray_writes, make_output


@pytest.mark.flash_attention_forward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_splitkv_mha_output_layout(monkeypatch, layout, dtype):
    batch, seq, heads, dim, kv_seq = 2, 17, 3, 96, 256
    generator = torch.Generator().manual_seed(23)
    q_cpu = torch.randn((batch, seq, heads, dim), generator=generator).to(dtype)
    k_cpu = torch.randn((batch, kv_seq, heads, dim), generator=generator).to(dtype)
    v_cpu = torch.randn((batch, kv_seq, heads, dim), generator=generator).to(dtype)
    q, k, v = [x.to(flag_gems.device) for x in (q_cpu, k_cpu, v_cpu)]
    out, backing, occupied = make_output(batch, seq, heads, dim, dtype, layout)

    tasks = batch * heads * triton.cdiv(seq, flash_api.block_m_splitkv_heuristic(dim))
    get_properties = flash_api.torch_device_fn.get_device_properties

    class DeviceProperties:
        # Force at least two KV splits without depending on the device SM count.
        multi_processor_count = tasks * 2

        def __init__(self, properties):
            self.properties = properties

        def __getattr__(self, name):
            return getattr(self.properties, name)

    monkeypatch.setattr(
        flash_api.torch_device_fn,
        "get_device_properties",
        lambda *args, **kwargs: DeviceProperties(get_properties(*args, **kwargs)),
    )

    observed_splits = []
    original = flash_api.flash_fwd_splitkv_combine_kernel

    class ObserveCombine:
        def __getitem__(self, grid):
            def run(**kwargs):
                observed_splits.append(kwargs["n_splits"])
                return original[grid](**kwargs)

            return run

    monkeypatch.setattr(
        flash_api, "flash_fwd_splitkv_combine_kernel", ObserveCombine()
    )

    result = flash_api.mha_fwd(
        q, k, v, out, None, 0.0, dim**-0.5, False, -1, -1, 0.0, False
    )
    assert observed_splits and all(n > 1 for n in observed_splits)

    scores = torch.einsum("bqhd,bkhd->bhqk", q_cpu.float(), k_cpu.float()) * dim**-0.5
    expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v_cpu.float())
    torch.testing.assert_close(
        result[0].cpu(), expected.to(dtype), atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(
        result[4].cpu(), scores.logsumexp(-1), atol=0.02, rtol=0.02
    )
    assert result[0].data_ptr() == out.data_ptr()
    assert_no_stray_writes(backing, occupied)
