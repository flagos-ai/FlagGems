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

import importlib.machinery
import importlib.util
import uuid
from pathlib import Path

import pytest


CODE_UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "flag_gems"
    / "utils"
    / "code_utils.py"
)


def load_code_utils():
    module_name = f"_generated_code_utils_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, CODE_UTILS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rebuilds_and_imports_after_nul_bytes(tmp_path, monkeypatch):
    code_utils = load_code_utils()
    target = tmp_path / "generated.py"
    source = "VALUE = 42\n\ndef _index_linearized_wrapper():\n    return VALUE\n"
    write_count = 0
    real_write_atomic = code_utils.write_atomic

    def corrupt_once(path, content, *args, **kwargs):
        nonlocal write_count
        write_count += 1
        real_write_atomic(path, content, *args, **kwargs)
        if write_count == 1:
            cached = Path(path)
            cached.write_bytes(b"\x00" * len(cached.read_bytes()))

    monkeypatch.setattr(code_utils, "write_atomic", corrupt_once)
    generated = code_utils.load_generated_module(
        "_generated_code_cache_recovery", target, source
    )

    assert generated._index_linearized_wrapper() == 42
    assert write_count == 2
    assert target.read_bytes() == source.encode("utf-8")


def test_rebuilds_after_nul_syntax_error_during_import(tmp_path, monkeypatch):
    code_utils = load_code_utils()
    target = tmp_path / "generated.py"
    source = "VALUE = 42\n\ndef _index_wrapper():\n    return VALUE\n"
    spec_calls = 0
    real_spec_from_file_location = code_utils.importlib.util.spec_from_file_location

    class NulByteLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise SyntaxError("source code string cannot contain null bytes")

    def fail_import_once(module_name, file_path):
        nonlocal spec_calls
        spec_calls += 1
        if spec_calls == 1:
            return importlib.machinery.ModuleSpec(module_name, NulByteLoader())
        return real_spec_from_file_location(module_name, file_path)

    monkeypatch.setattr(
        code_utils.importlib.util, "spec_from_file_location", fail_import_once
    )
    generated = code_utils.load_generated_module(
        "_generated_code_cache_import_nul", target, source
    )

    assert generated._index_wrapper() == 42
    assert spec_calls == 2


def test_persistent_nul_fails_after_one_retry(tmp_path, monkeypatch):
    code_utils = load_code_utils()
    target = tmp_path / "generated.py"
    write_count = 0
    real_write_atomic = code_utils.write_atomic

    def always_corrupt(path, content, *args, **kwargs):
        nonlocal write_count
        write_count += 1
        real_write_atomic(path, content, *args, **kwargs)
        cached = Path(path)
        cached.write_bytes(b"\x00" * len(cached.read_bytes()))

    monkeypatch.setattr(code_utils, "write_atomic", always_corrupt)

    with pytest.raises(RuntimeError, match="Generated code cache remained corrupt.*FLAGGEMS_CACHE_DIR"):
        code_utils.load_generated_module(
            "_generated_code_cache_persistent_nul", target, "VALUE = 42\n"
        )

    assert write_count == 2
    assert b"\x00" in target.read_bytes()


def test_source_with_nul_fails_before_write(tmp_path, monkeypatch):
    code_utils = load_code_utils()
    target = tmp_path / "generated.py"
    write_count = 0
    real_write_atomic = code_utils.write_atomic

    def count_writes(path, content, *args, **kwargs):
        nonlocal write_count
        write_count += 1
        real_write_atomic(path, content, *args, **kwargs)

    monkeypatch.setattr(code_utils, "write_atomic", count_writes)

    with pytest.raises(RuntimeError, match="source contains NUL bytes"):
        code_utils.load_generated_module(
            "_generated_code_cache_source_nul", target, "VALUE = \"\x00\"\n"
        )

    assert write_count == 0
    assert not target.exists()


def test_non_nul_syntax_error_is_not_masked(tmp_path, monkeypatch):
    code_utils = load_code_utils()
    target = tmp_path / "generated.py"
    write_count = 0
    real_write_atomic = code_utils.write_atomic

    def count_writes(path, content, *args, **kwargs):
        nonlocal write_count
        write_count += 1
        real_write_atomic(path, content, *args, **kwargs)

    monkeypatch.setattr(code_utils, "write_atomic", count_writes)

    with pytest.raises(SyntaxError):
        code_utils.load_generated_module(
            "_generated_code_cache_syntax", target, "def invalid(:\n"
        )

    assert write_count == 1
