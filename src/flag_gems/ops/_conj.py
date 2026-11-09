# Copyright 2026, The FlagOS Contributors.
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
#
"""FlagGems ``_conj``: the zero-copy conjugate view of ``self``.

``aten::_conj`` is a pure metadata operator. It returns a new tensor that
shares the input's storage, with the lazy conjugate bit toggled. Nothing is
read, moved or written: the same buffer is merely interpreted with the
imaginary component negated.

Contract points that drive the implementation:

* **Never materialize.** ``conj_physical`` / ``resolve_conj`` / ``clone`` /
  ``contiguous`` / an eager elementwise pass would allocate and copy, turning
  an O(1) metadata op into an O(N) memory stream. That also breaks the
  documented view semantics (shared storage, mutation visibility) and doubles
  the peak memory footprint.
* **Toggle, don't just set.** ``conj(conj(x))`` must clear the bit and expose
  the original logical value; ATen's implementation handles both directions.
* **Preserve autograd.** The native ``_conj`` view carries ``ConjBackward0``
  view-replay metadata. Flipping the bit by hand with ``_set_conj`` would lose
  it and produce a gradient with the wrong sign.

The only host-side work here is the dispatch guard. FlagGems registers
``_conj`` at the backend dispatch key, so a naive ``torch._conj`` call would
re-enter this module. Excluding the backend key for this one call lets ATen's
generic CompositeExplicitAutograd implementation (which is backend-agnostic)
build the view instead.
"""

import logging

import torch

from flag_gems import runtime

logger = logging.getLogger(__name__)

# NOTE: parse_dispatch_key also resolves renamed PrivateUse1 keys (e.g. MUSA).
_BACKEND_KEYSET = torch._C.DispatchKeySet(
    torch._C._parse_dispatch_key(runtime.device.dispatch_key)
)


def _conj(input: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS CONJ")
    if not input.is_complex():
        raise RuntimeError("_conj only supports complex tensors")

    # Bypass FlagGems' own backend registration: native `_conj` returns the
    # lazy conjugate view (zero copy, shared storage, ConjBackward0).
    with torch._C._ExcludeDispatchKeyGuard(_BACKEND_KEYSET):
        return torch._conj(input)
