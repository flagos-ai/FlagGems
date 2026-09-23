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

from flag_gems.patches import patch_util


@pytest.mark.parametrize(
    ("lib_name", "module_names"),
    [
        ("_C", ("vllm._C", "vllm._C_stable_libtorch")),
        ("_moe_C", ("vllm._moe_C", "vllm._moe_C_stable_libtorch")),
        ("_vllm_fa3_C", ("vllm.vllm_flash_attn._vllm_fa3_C",)),
        ("_C_cache_ops", ("vllm._C_cache_ops", "vllm._C_stable_libtorch")),
    ],
)
def test_vllm_extension_legacy_and_stable_abi_fallback(
    monkeypatch, lib_name, module_names
):
    attempts = []

    def try_import(module_name):
        attempts.append(module_name)
        return module_name == module_names[-1]

    monkeypatch.setattr(patch_util, "_try_import_vllm_extension", try_import)

    assert patch_util._ensure_vllm_library_exists(lib_name)
    assert attempts == list(module_names)
