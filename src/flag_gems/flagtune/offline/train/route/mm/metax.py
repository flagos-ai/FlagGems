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

"""MetaX MM route selection used by the shared FlagTune resolver."""

from typing import Any


def select_mm_route(a: Any, b: Any, module: Any) -> str:
    """Use the public MM dispatch without launching or copying input tensors."""
    if a.shape[1] != b.shape[0]:
        return "invalid"
    m, k = a.shape
    _, n = b.shape
    if not m or not n or not k:
        return "empty"
    if m == 1 or n == 1:
        return "metax_mv"

    a_strides, b_strides = a.stride(), b.stride()
    properties = module.get_device_properties(a.device.index)
    kernel, split_k, pack_rhs = module._dispatch_mm(
        m,
        n,
        k,
        a_strides,
        b_strides,
        (n, 1),
        a.dtype,
        a.dtype,
        all(t.data_ptr() % module._VECTOR_ALIGNMENT_BYTES == 0 for t in (a, b)),
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape[::-1]
        and a_strides == b_strides[::-1],
        properties.multi_processor_count,
        properties.shared_memory_per_block,
        properties.L2_cache_size,
    )
    # Fresh MM outputs are contiguous and aligned. The new kernel signatures
    # must not bind to the published metax_nn/nt/gemv cost models.
    route = "metax_" + kernel.jit_function.__name__.lstrip("_")
    if split_k > 1:
        route += "_splitk"
    if pack_rhs:
        route += "_packed"
    return route
