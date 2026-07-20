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

import pytest

from flag_gems.runtime import backend


def test_non_nvidia_heuristic_import_failure_does_not_fallback(monkeypatch):
    imported_modules = []
    import_module = backend.importlib.import_module

    def fake_import_module(module_name):
        imported_modules.append(module_name)
        if module_name == "_mthreads.heuristics_config_utils":
            raise ModuleNotFoundError("simulated mthreads heuristic import failure")
        return import_module(module_name)

    monkeypatch.setattr(backend.importlib, "import_module", fake_import_module)

    with pytest.raises(ModuleNotFoundError, match="mthreads heuristic"):
        backend.get_heuristic_config("mthreads")

    assert "_nvidia.heuristics_config_utils" not in imported_modules


def test_mthreads_softmax_non_inner_uses_mthreads_heuristic():
    configs = backend.get_heuristic_config("mthreads")
    tile_k = configs["softmax_non_inner"]["TILE_K"]

    assert tile_k.__module__ == "_mthreads.heuristics_config_utils"
