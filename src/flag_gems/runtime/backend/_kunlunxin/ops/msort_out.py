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

"""Kunlunxin ``msort.out`` override re-export.

The implementation lives next to the retained ``msort`` kernels in
``.msort``.  This thin module only re-exports it so the shared
``ops/__init__.py`` can be wired with
``harness/tools/wire_override.py --module msort_out --names msort_out``
(the helper appends a new import line only for modules that are not imported
yet, so an incremental name added to an already-imported module needs its own
module entry point).
"""

from .msort import msort_out  # noqa: F401

__all__ = ["msort_out"]