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

"""gcu400 real: reuse the generic zero-copy view implementation.

The generic ``real`` is a pure metadata path (``view_as_real`` + ``select``)
and is correct on this backend.  A verbatim copy here would be installed on
``flag_gems.real`` by the arch-ops override as a second function object,
detaching the public entry from the generic module globals -- callers that
instrument ``flag_gems.ops.real`` (e.g. its ``_VIEW_AS_REAL`` view path) would
observe nothing.  Re-exporting the generic functions keeps a single
implementation for the direct call path and the aten registration.
"""

from flag_gems.ops.real import real, real_conjugate, real_device

__all__ = ["real", "real_conjugate", "real_device"]
