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
    c = module.torch.empty((m, n), device=a.device, dtype=a.dtype)
    plan = module._dispatch_mm(module._features(a, b, c))
    # The new kernels have different signatures and configuration spaces from
    # the published metax_nn/nt/gemv models. Report their actual routes so the
    # shared resolver marks them unadapted instead of binding a legacy model.
    route = "metax_" + plan.launch.__name__.removeprefix("_launch_")
    if plan.split_k > 1:
        route += "_splitk"
    if plan.pack_rhs:
        route += "_packed"
    return route
