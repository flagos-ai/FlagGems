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

import flag_gems
from flag_gems import _FULL_CONFIG

from . import accuracy_utils as utils

# Representative host buffers used for asynchronous host-to-device copies.
PIN_MEMORY_SHAPES = [(1024,), (1024, 1024), (4096, 4096)]


@pytest.mark.underscore_pin_memory
@pytest.mark.parametrize("shape", PIN_MEMORY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__pin_memory(shape, dtype):
    # _pin_memory only accepts CPU tensors, so the input always lives on CPU.
    inp = torch.randn(shape, dtype=dtype, device="cpu")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._pin_memory(ref_inp)

    res_out = flag_gems._pin_memory(inp)

    assert res_out.is_pinned()
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.underscore_pin_memory
def test__pin_memory_registered_on_cpu_key():
    """``_pin_memory`` must be registered under the CPU dispatch key.

    Its input is always a CPU tensor and the native op is
    CompositeExplicitAutograd, so it dispatches on CPU rather than on the
    accelerator key. Registering it under the backend key leaves the native
    kernel in charge and the impl never runs. The accuracy test above calls the
    implementation directly, so it passes either way and cannot catch this.
    """
    entry = next(e for e in _FULL_CONFIG if e[0] == "_pin_memory")

    assert len(entry) == 4, "_pin_memory needs an explicit dispatch key list"
    assert list(entry[3]) == ["CPU"]
