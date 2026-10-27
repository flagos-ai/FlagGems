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

import torch

from flag_gems.ops.index_put import _index_put_impl_ as _generic_index_put_impl_
from flag_gems.ops.index_put import index_put as _generic_index_put
from flag_gems.ops.index_put import index_put_ as _generic_index_put_

_LARGE_INDEX_PUT_INPUT_NUMEL = 2**26


def _use_native_index_put(inp: torch.Tensor) -> bool:
    return inp.numel() >= _LARGE_INDEX_PUT_INPUT_NUMEL


def _cpu_index_put(inp, indices, values, accumulate):
    cpu_indices = [index.to("cpu") if index is not None else None for index in indices]
    cpu_inp = inp.to("cpu")
    cpu_values = values.to("cpu")
    return torch.ops.aten.index_put.default(
        cpu_inp, cpu_indices, cpu_values, accumulate
    ).to(inp.device)


def _cpu_index_put_(inp, indices, values, accumulate):
    result = _cpu_index_put(inp, indices, values, accumulate)
    inp.copy_(result)
    return inp


def index_put(inp, indices, values, accumulate=False):
    if _use_native_index_put(inp):
        return _cpu_index_put(inp, indices, values, accumulate)
    return _generic_index_put(inp, indices, values, accumulate)


def index_put_(inp, indices, values, accumulate=False):
    if _use_native_index_put(inp):
        return _cpu_index_put_(inp, indices, values, accumulate)
    return _generic_index_put_(inp, indices, values, accumulate)


def _index_put_impl_(inp, indices, values, accumulate=False, unsafe=False):
    if _use_native_index_put(inp):
        del unsafe
        return _cpu_index_put_(inp, indices, values, accumulate)
    return _generic_index_put_impl_(inp, indices, values, accumulate, unsafe)
