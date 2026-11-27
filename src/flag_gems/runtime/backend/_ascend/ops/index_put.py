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

"""Ascend-specialized ``index_put`` / ``index_put_`` with a native fallback.

The Triton implementation crashes in the Ascend MLIR compiler for common
mask/index combinations: issue #4280 (``tensor<2x1xi1>`` andi lowering in the
``TritonToUnstructureIncubated`` pass) and issue #4020 (``llvm::cast<Attribute>``
assertion in the ``ttir_to_linalg`` pass). Both are compiler bugs, not kernel
bugs, so until they are fixed this backend routes ``index_put`` / ``index_put_``
to the PyTorch native operator, which lowers through PTA/torch_npu instead of
the MLIR pipeline.

Set ``FLAGGEMS_ASCEND_INDEX_PUT_TRITON=1`` to force the original Triton path
back on, e.g. to re-check correctness and performance once the compiler fixes
land. Unset (the default) keeps the native fallback.
"""

import logging
import os

import torch

from flag_gems.ops.index_put import index_put as _triton_index_put
from flag_gems.ops.index_put import index_put_ as _triton_index_put_

logger = logging.getLogger(__name__)

_FORCE_TRITON_ENV = "FLAGGEMS_ASCEND_INDEX_PUT_TRITON"

# Redispatch the native ATen kernel, bypassing our own registered dispatch:
# calling ``inp.index_put_(...)`` directly would recurse straight back into this
# override (same reason _ascend/ops/log_.py raises instead of delegating).
_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)

# index_put is a high-frequency op, so warn about the implementation switch only
# once per process instead of on every call. WARNING (not INFO) so the switch is
# visible at the default log level.
_LOGGED_NATIVE_FALLBACK = False


def _use_triton_path() -> bool:
    """True when the caller opted back into the Triton implementation."""
    return os.getenv(_FORCE_TRITON_ENV, "0").lower() in ("1", "true", "on")


def _native_index_put(inp, indices, values, accumulate, inplace):
    global _LOGGED_NATIVE_FALLBACK
    if not _LOGGED_NATIVE_FALLBACK:
        logger.warning(
            "ascend: index_put/index_put_ falling back to native ATen "
            "(set FLAGGEMS_ASCEND_INDEX_PUT_TRITON=1 to override)"
        )
        _LOGGED_NATIVE_FALLBACK = True
    indices = list(indices)
    if values.device != inp.device:
        values = values.to(inp.device)
    if inplace:
        return torch.ops.aten.index_put_.default.redispatch(
            _FALLBACK_KEYSET, inp, indices, values, accumulate
        )
    return torch.ops.aten.index_put.default.redispatch(
        _FALLBACK_KEYSET, inp, indices, values, accumulate
    )


def index_put(inp, indices, values, accumulate=False):
    if _use_triton_path():
        logger.debug("GEMS_ASCEND INDEX_PUT (triton)")
        return _triton_index_put(inp, indices, values, accumulate)
    logger.debug("GEMS_ASCEND INDEX_PUT (native fallback)")
    return _native_index_put(inp, indices, values, accumulate, inplace=False)


def index_put_(inp, indices, values, accumulate=False):
    if _use_triton_path():
        logger.debug("GEMS_ASCEND INDEX_PUT_ (triton)")
        return _triton_index_put_(inp, indices, values, accumulate)
    logger.debug("GEMS_ASCEND INDEX_PUT_ (native fallback)")
    return _native_index_put(inp, indices, values, accumulate, inplace=True)
