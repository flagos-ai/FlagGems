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

"""Atlas A5 (Ascend950) operator specializations.

Add one module per operator (``a5/ops/<op_name>.py``, same file and function
name as the operator it overrides) and export it here:

    from .add import add

    __all__ = [
        "add",
    ]

``BackendArchEvent.get_arch_ops`` collects every function in this module's
namespace, so only operator entry points belong in ``__all__``; keep helpers
private (leading underscore) or inside their own module. ``__all__`` must stay
sorted by ``str.casefold`` (``tools/ci_checks/check_init_exports.py``).

See ``a5/__init__.py`` for the import rules that apply to modules here.
"""

from .cat import cat

__all__ = ["cat"]
