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

"""Guards must compare against the device name tensors actually report (#7135)."""

import types

import pytest
import torch

from flag_gems.runtime.backend.device_finder import resolve_guard_device_name


@pytest.fixture
def cleanup_env(monkeypatch):
    monkeypatch.delenv("GEMS_DEVICE_NAME", raising=False)
    monkeypatch.delenv("FLAGGEMS_DEVICE_NAME", raising=False)
    return monkeypatch


def test_keeps_declared_name_when_backend_is_available(cleanup_env):
    cleanup_env.setattr(
        torch, "cuda", types.SimpleNamespace(is_available=lambda: True)
    )
    assert resolve_guard_device_name("cuda") == "cuda"


def test_uses_registered_privateuse1_name(cleanup_env):
    # torch_fl registers the accelerator as ``flagos`` while the descriptor
    # still declares ``cuda``; ops compare against the former.
    cleanup_env.setattr(
        torch, "cuda", types.SimpleNamespace(is_available=lambda: False)
    )
    cleanup_env.setattr(torch._C, "_get_privateuse1_backend_name", lambda: "flagos")
    cleanup_env.setattr(torch, "flagos", types.SimpleNamespace(), raising=False)
    assert resolve_guard_device_name("cuda") == "flagos"


def test_env_override_wins(cleanup_env):
    cleanup_env.setenv("GEMS_DEVICE_NAME", "flagos")
    assert resolve_guard_device_name("cuda") == "flagos"


def test_falls_back_to_declared_name_without_privateuse1(cleanup_env):
    cleanup_env.setattr(
        torch, "cuda", types.SimpleNamespace(is_available=lambda: False)
    )
    cleanup_env.setattr(
        torch._C, "_get_privateuse1_backend_name", lambda: "privateuseone"
    )
    assert resolve_guard_device_name("cuda") == "cuda"
