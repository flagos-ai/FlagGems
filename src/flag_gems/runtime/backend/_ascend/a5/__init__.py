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

"""Atlas A5 (Ascend950DT / Ascend950PR, NpuArch DAV_3510) specializations.

``_ascend/__init__.py`` resolves the arch key ``a5`` in ``get_arch_key`` when
``ARCH`` says so or the driver reports an Ascend950 SoC, and
``runtime.backend.BackendArchEvent`` then loads
this directory and overrides the matching entries of ``_ascend/ops`` (A2/A3
baseline) and ``flag_gems.ops`` (generic). Every operator placed under
``a5/ops`` and exported from ``a5/ops/__init__.py`` wins over both on A5 only;
nothing here is reachable on Ascend910B*.

Optional per-arch files, all picked up by ``BackendArchEvent`` when present:

* ``tune_configs.yaml`` -- ``triton.autotune`` configs, consulted before
  ``_ascend/tune_configs.yaml``.
* ``heuristics_config_utils.py`` -- must define ``HEURISTICS_CONFIGS``, which
  has to be re-exported from this file to be visible to the loader.
* ``enable_configs.yaml`` -- per-arch operator include/exclude lists.

Note on imports: this package is imported under its bare name ``a5`` (via
``sys.path`` injection), not as ``flag_gems.runtime.backend._ascend.a5``, so it
has no parent package at runtime. Modules under ``a5/ops`` must reach shared
code through absolute imports (``from flag_gems.runtime.backend._ascend import
heuristics_config_utils``, ``from flag_gems.utils import libentry``); a
relative ``from ..utils import CORE_NUM`` raises ImportError.

Keep this file free of operator imports. ``BackendArchEvent`` imports it while
``flag_gems.runtime`` is still initializing (``configs_loader`` builds the
event to read the arch configs), so pulling in ``a5/ops`` from here would
import ``flag_gems.ops``/``flag_gems.runtime`` mid-initialization and fail with
a circular import. The loader imports ``a5.ops`` later, on its own.
"""
