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


def _dispatch(a: Any, b: Any, module: Any):
    m, k = a.shape
    n = b.shape[1]
    a_strides, b_strides = a.stride(), b.stride()
    properties = module.get_device_properties(a.device.index)
    return module._dispatch_mm(
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


def select_mm_route(a: Any, b: Any, module: Any) -> str:
    if a.shape[1] != b.shape[0]:
        return "invalid"
    m, k = a.shape
    n = b.shape[1]
    if not m or not n or not k:
        return "empty"
    if m == 1 or n == 1:
        # MM delegates these cases to MV, which has no MM model binding.
        return "metax_mv"
    kernel, split_k, _pack_rhs = _dispatch(a, b, module)
    if split_k > 1:
        kernel = module._PARTIAL_KERNELS[kernel]
    return kernel.fn._flagtune_variant


def dynamic_inputs(a: Any, b: Any, module: Any) -> dict[str, Any]:
    _, split_k, pack_rhs = _dispatch(a, b, module)
    return {"SPLIT_K": split_k, "PACK_RHS": pack_rhs}
