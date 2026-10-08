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

"""
Unit tests for dynamic operator override functionality.

Tests the DynamicOpOverride class from flag_gems.dynamic_registry and
the CLI integration from flag_gems.cli_override, verifying:

- Basic override and restore operations
- Loading implementations from files
- Batch override operations
- Context manager behavior
- CLI argument parsing and application
- pytest integration via --override and --override-config options
"""

import ast
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

import flag_gems
from flag_gems.dynamic_registry import DynamicOpOverride

# Needed for TestPytestIntegration, which uses the `pytester` fixture to
# run pytest itself as a subprocess and verify the --override/--override-config
# CLI options work end-to-end through the real pytest hooks.
pytest_plugins = ["pytester"]

# Repo root, used by TestRegistrarLiveOverrideResolution to load source
# files as throwaway submodules of a fake `flag_gems` package, and to spawn
# subprocess pytest runs, without requiring a GPU or a real Torch import.
ROOT = Path(__file__).resolve().parents[2]


def test_override_call_count_snapshots_and_all_skip_restore(tmp_path):
    source = tmp_path / "candidate.py"
    source.write_text("def run(value): return value\n")
    registry = DynamicOpOverride()
    registry.override_from_file("neg", str(source), "run")
    snapshot = registry.call_counts()
    snapshot["flag_gems.neg"] = 100
    assert registry.call_counts()["flag_gems.neg"] == 0
    registry.get_override("neg")(1)
    assert registry.call_counts()["flag_gems.neg"] == 1
    registry.restore_all()
    assert registry.call_counts() == {}
    registry.override_from_file("neg", str(source), "run")
    registry.restore_all(allow_unused=True)
    assert registry.list_overrides() == []


# Test fixtures for custom implementations
def custom_abs_impl(input):
    """Custom implementation of abs that adds a marker"""
    result = torch.abs(input)
    # Add a custom attribute to verify this implementation was called
    result._custom_marker = "custom_abs_called"
    return result


def custom_neg_impl(input):
    """Custom implementation of neg"""
    result = torch.neg(input)
    result._custom_marker = "custom_neg_called"
    return result


def custom_add_impl(input, other, *, alpha=1):
    """Custom implementation of add"""
    result = torch.add(input, other, alpha=alpha)
    result._custom_marker = "custom_add_called"
    return result


class TestDynamicOpOverride:
    """Test suite for DynamicOpOverride class"""

    def test_basic_override(self):
        """Test basic operator override functionality"""
        registry = DynamicOpOverride()

        # Override abs operator
        success = registry.override("abs", custom_abs_impl)
        assert success, "Override should succeed"

        # Test that custom implementation is used
        x = torch.tensor([-1.0, -2.0, 3.0], device=flag_gems.device)
        result = flag_gems.abs(x)

        assert hasattr(
            result, "_custom_marker"
        ), "Custom implementation should be called"
        assert result._custom_marker == "custom_abs_called"

        # Verify correctness
        expected = torch.abs(x)
        torch.testing.assert_close(result, expected)

        # Cleanup
        registry.restore("abs")

    def test_multiple_overrides(self):
        """Test overriding multiple operators"""
        registry = DynamicOpOverride()

        # Override multiple operators
        registry.override("abs", custom_abs_impl)
        registry.override("neg", custom_neg_impl)

        # Test both
        x = torch.tensor([1.0, -2.0, 3.0], device=flag_gems.device)

        abs_result = flag_gems.abs(x)
        assert hasattr(abs_result, "_custom_marker")
        assert abs_result._custom_marker == "custom_abs_called"

        neg_result = flag_gems.neg(x)
        assert hasattr(neg_result, "_custom_marker")
        assert neg_result._custom_marker == "custom_neg_called"

        # Cleanup
        registry.restore_all()

    def test_restore_single_op(self):
        """Test restoring a single overridden operator"""
        registry = DynamicOpOverride()

        # Store original
        original_abs = flag_gems.abs

        # Override
        registry.override("abs", custom_abs_impl)
        assert flag_gems.abs != original_abs

        # Restore
        success = registry.restore("abs")
        assert success, "Restore should succeed"
        assert flag_gems.abs == original_abs, "Should restore to original"

        # Test that original is used
        x = torch.tensor([-1.0, -2.0], device=flag_gems.device)
        result = flag_gems.abs(x)
        assert not hasattr(result, "_custom_marker")

    def test_restore_all(self):
        """Test restoring all overridden operators"""
        registry = DynamicOpOverride()

        original_abs = flag_gems.abs
        original_neg = flag_gems.neg

        # Override multiple
        registry.override("abs", custom_abs_impl)
        registry.override("neg", custom_neg_impl)

        # Restore all
        registry.restore_all()

        assert flag_gems.abs == original_abs
        assert flag_gems.neg == original_neg

    def test_context_manager(self):
        """Test using DynamicOpOverride as context manager"""
        original_abs = flag_gems.abs

        with DynamicOpOverride() as registry:
            registry.override("abs", custom_abs_impl)

            x = torch.tensor([-1.0], device=flag_gems.device)
            result = flag_gems.abs(x)
            assert hasattr(result, "_custom_marker")

        # Should be restored after exiting context
        assert flag_gems.abs == original_abs

    def test_list_overrides(self):
        """Test listing active overrides"""
        registry = DynamicOpOverride()

        assert len(registry.list_overrides()) == 0

        registry.override("abs", custom_abs_impl)
        overrides = registry.list_overrides()
        assert len(overrides) == 1
        assert "flag_gems.abs" in overrides

        registry.override("neg", custom_neg_impl)
        overrides = registry.list_overrides()
        assert len(overrides) == 2

        registry.restore_all()
        assert len(registry.list_overrides()) == 0

    def test_override_from_file(self):
        """Test loading implementation from a file"""
        # Create a temporary file with custom implementation
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
import torch

def my_custom_abs(input):
    result = torch.abs(input)
    result._file_marker = "loaded_from_file"
    return result
""")
            temp_file = f.name

        try:
            registry = DynamicOpOverride()

            # Load from file
            success = registry.override_from_file("abs", temp_file, "my_custom_abs")
            assert success, "Should load from file successfully"

            # Test
            x = torch.tensor([-1.0, -2.0], device=flag_gems.device)
            result = flag_gems.abs(x)
            assert hasattr(result, "_file_marker")
            assert result._file_marker == "loaded_from_file"

            registry.restore_all()
        finally:
            Path(temp_file).unlink()

    def test_override_batch_from_files(self):
        """Test batch override from multiple files"""
        # Create temporary files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create abs implementation
            abs_file = tmpdir / "custom_abs.py"
            abs_file.write_text("""
import torch
def my_abs(input):
    result = torch.abs(input)
    result._marker = "abs_from_file"
    return result
""")

            # Create neg implementation
            neg_file = tmpdir / "custom_neg.py"
            neg_file.write_text("""
import torch
def my_neg(input):
    result = torch.neg(input)
    result._marker = "neg_from_file"
    return result
""")

            registry = DynamicOpOverride()

            # Batch override
            results = registry.override_batch_from_files(
                {
                    "abs": (str(abs_file), "my_abs"),
                    "neg": (str(neg_file), "my_neg"),
                }
            )

            assert results["abs"], "abs override should succeed"
            assert results["neg"], "neg override should succeed"

            # Test both
            x = torch.tensor([1.0, -2.0], device=flag_gems.device)

            abs_result = flag_gems.abs(x)
            assert hasattr(abs_result, "_marker")
            assert abs_result._marker == "abs_from_file"

            neg_result = flag_gems.neg(x)
            assert hasattr(neg_result, "_marker")
            assert neg_result._marker == "neg_from_file"

            registry.restore_all()

    def test_override_nonexistent_file(self):
        """Test handling of nonexistent file"""
        registry = DynamicOpOverride()

        success = registry.override_from_file(
            "abs", "/nonexistent/path/file.py", "some_func"
        )

        assert not success, "Should fail for nonexistent file"

    def test_override_missing_function(self):
        """Test handling of missing function in file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def some_other_function():
    pass
""")
            temp_file = f.name

        try:
            registry = DynamicOpOverride()

            success = registry.override_from_file(
                "abs", temp_file, "nonexistent_function"
            )

            assert not success, "Should fail for missing function"
        finally:
            Path(temp_file).unlink()

    def test_restore_without_override(self):
        """Test restoring an operator that wasn't overridden"""
        registry = DynamicOpOverride()

        # Try to restore without overriding first
        success = registry.restore("abs")
        assert not success, "Should fail when operator wasn't overridden"

    def test_operator_with_kwargs(self):
        """Test overriding operators with keyword arguments"""
        registry = DynamicOpOverride()

        registry.override("add", custom_add_impl)

        x = torch.tensor([1.0, 2.0], device=flag_gems.device)
        y = torch.tensor([3.0, 4.0], device=flag_gems.device)

        # Test with alpha parameter
        result = flag_gems.add(x, y, alpha=2.0)
        assert hasattr(result, "_custom_marker")
        assert result._custom_marker == "custom_add_called"

        expected = x + 2.0 * y
        torch.testing.assert_close(result, expected)

        registry.restore_all()

    def test_concurrent_registries(self):
        """Test multiple independent registries"""
        registry1 = DynamicOpOverride()

        original_abs = flag_gems.abs

        # Override with registry1
        registry1.override("abs", custom_abs_impl)
        assert flag_gems.abs != original_abs

        # Registry2 should see the override from registry1
        # (because they modify the same module)
        x = torch.tensor([-1.0], device=flag_gems.device)
        result = flag_gems.abs(x)
        assert hasattr(result, "_custom_marker")

        # Restore with registry1
        registry1.restore("abs")
        assert flag_gems.abs == original_abs

    def test_override_special_ops(self):
        """Test overriding operators with special names"""

        def custom_list_to_tensor(list_of_ints, dtype=None, device=None):
            """Custom implementation of _list_to_tensor"""
            result = torch.tensor(list_of_ints, dtype=dtype, device=device)
            result._custom_marker = "custom_list_to_tensor"
            return result

        registry = DynamicOpOverride()

        # Test with underscore-prefixed operator
        registry.override("_list_to_tensor", custom_list_to_tensor)

        result = flag_gems._list_to_tensor([1, 2, 3], device=flag_gems.device)
        assert hasattr(result, "_custom_marker")
        assert result._custom_marker == "custom_list_to_tensor"

        registry.restore_all()


class TestCLIOverride:
    """Test suite for CLI override functionality"""

    def test_parse_override_spec_colon(self):
        """Test parsing override spec with colon separator"""
        from flag_gems.cli_override import parse_override_spec

        # Format: op_name:filepath
        op_name, filepath, func_name = parse_override_spec("softmax:./custom.py")
        assert op_name == "softmax"
        assert filepath == "./custom.py"
        assert func_name == "softmax"

        # Format: op_name:filepath:func_name
        op_name, filepath, func_name = parse_override_spec(
            "softmax:./custom.py:my_softmax"
        )
        assert op_name == "softmax"
        assert filepath == "./custom.py"
        assert func_name == "my_softmax"

    def test_parse_override_spec_equals(self):
        """Test parsing override spec with equals separator"""
        from flag_gems.cli_override import parse_override_spec

        # Format: op_name=filepath
        op_name, filepath, func_name = parse_override_spec("softmax=./custom.py")
        assert op_name == "softmax"
        assert filepath == "./custom.py"
        assert func_name == "softmax"

        # Format: op_name=filepath:func_name
        op_name, filepath, func_name = parse_override_spec(
            "softmax=./custom.py:my_softmax"
        )
        assert op_name == "softmax"
        assert filepath == "./custom.py"
        assert func_name == "my_softmax"

    def test_parse_override_spec_invalid(self):
        """Test parsing invalid override spec"""
        from flag_gems.cli_override import parse_override_spec

        with pytest.raises(ValueError):
            parse_override_spec("invalid_spec_without_separator")

    def test_load_override_config_yaml(self):
        """Test loading override config from YAML file"""
        from flag_gems.cli_override import load_override_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
overrides:
  softmax:
    file: ./custom_softmax.py
    function: my_softmax

  rms_norm:
    file: ./custom_rms_norm.py

  layer_norm: ./custom_layer_norm.py

  gelu: ./custom_gelu.py:my_gelu
""")
            temp_file = f.name

        try:
            overrides = load_override_config(temp_file)

            assert "softmax" in overrides
            assert overrides["softmax"] == ("./custom_softmax.py", "my_softmax")

            assert "rms_norm" in overrides
            assert overrides["rms_norm"] == ("./custom_rms_norm.py", "rms_norm")

            assert "layer_norm" in overrides
            assert overrides["layer_norm"] == ("./custom_layer_norm.py", "layer_norm")

            assert "gelu" in overrides
            assert overrides["gelu"] == ("./custom_gelu.py", "my_gelu")
        finally:
            Path(temp_file).unlink()

    def test_load_override_config_json(self):
        """Test loading override config from JSON file"""
        from flag_gems.cli_override import load_override_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("""
{
  "overrides": {
    "softmax": {
      "file": "./custom_softmax.py",
      "function": "my_softmax"
    },
    "rms_norm": "./custom_rms_norm.py"
  }
}
""")
            temp_file = f.name

        try:
            overrides = load_override_config(temp_file)

            assert "softmax" in overrides
            assert overrides["softmax"] == ("./custom_softmax.py", "my_softmax")

            assert "rms_norm" in overrides
            assert overrides["rms_norm"] == ("./custom_rms_norm.py", "rms_norm")
        finally:
            Path(temp_file).unlink()

    def test_load_override_config_missing_file(self):
        """Test loading config from nonexistent file"""
        from flag_gems.cli_override import load_override_config

        with pytest.raises(FileNotFoundError):
            load_override_config("/nonexistent/config.yaml")


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios"""

    def test_ab_testing_scenario(self):
        """Test A/B testing two different implementations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create implementation A
            impl_a = tmpdir / "impl_a.py"
            impl_a.write_text("""
import torch
def softmax_impl(input, dim=-1, dtype=None):
    result = torch.softmax(input, dim=dim, dtype=dtype)
    result._variant = "A"
    return result
""")

            # Create implementation B
            impl_b = tmpdir / "impl_b.py"
            impl_b.write_text("""
import torch
def softmax_impl(input, dim=-1, dtype=None):
    result = torch.softmax(input, dim=dim, dtype=dtype)
    result._variant = "B"
    return result
""")

            x = torch.randn(10, 10, device=flag_gems.device)

            # Test variant A
            with DynamicOpOverride() as reg_a:
                reg_a.override_from_file("softmax", str(impl_a), "softmax_impl")
                result_a = flag_gems.softmax(x)
                assert result_a._variant == "A"

            # Test variant B
            with DynamicOpOverride() as reg_b:
                reg_b.override_from_file("softmax", str(impl_b), "softmax_impl")
                result_b = flag_gems.softmax(x)
                assert result_b._variant == "B"

            # Both should produce same numerical result
            torch.testing.assert_close(result_a, result_b)

    def test_progressive_override(self):
        """Test progressively adding overrides during testing"""
        registry = DynamicOpOverride()

        x = torch.tensor([1.0, -2.0], device=flag_gems.device)

        # Start with no overrides
        result = flag_gems.abs(x)
        assert not hasattr(result, "_custom_marker")

        # Add first override
        registry.override("abs", custom_abs_impl)
        result = flag_gems.abs(x)
        assert hasattr(result, "_custom_marker")

        # Add second override
        registry.override("neg", custom_neg_impl)
        result = flag_gems.neg(x)
        assert hasattr(result, "_custom_marker")

        # Restore one at a time
        registry.restore("abs")
        result = flag_gems.abs(x)
        assert not hasattr(result, "_custom_marker")

        result = flag_gems.neg(x)
        assert hasattr(result, "_custom_marker")  # Still overridden

        registry.restore_all()


class TestPytestIntegration:
    """
    End-to-end tests that invoke pytest as a subprocess (via the
    ``pytester`` fixture) to verify that ``--override``/``--override-config``
    work through the real ``pytest_addoption``/``pytest_configure``/
    ``pytest_unconfigure`` hooks wired up in ``tests/conftest.py`` and
    ``benchmark/conftest.py``.

    These tests exist to catch integration bugs (e.g. option-parsing API
    mismatches between ``argparse`` and pytest's own ``Parser``) that unit
    tests calling ``DynamicOpOverride`` directly cannot see, since those
    never go through ``pytest_addoption``.
    """

    @staticmethod
    def _repo_src_path() -> str:
        return str(Path(__file__).resolve().parents[2] / "src")

    def test_override_option_registered(self, pytester):
        """--override/--override-config/--list-overrides show up in --help."""
        pytester.makeini(f"""
            [pytest]
            pythonpath = {self._repo_src_path()}
            """)
        pytester.makeconftest("""
            from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

            def pytest_addoption(parser):
                add_override_arguments(parser)

            def pytest_configure(config):
                config._override_registry = apply_overrides_from_args(config.option)

            def pytest_unconfigure(config):
                if hasattr(config, "_override_registry"):
                    config._override_registry.restore_all()
            """)

        result = pytester.runpytest("--help")
        result.stdout.fnmatch_lines(["*--override=SPEC*"])
        result.stdout.fnmatch_lines(["*--override-config=PATH*"])
        result.stdout.fnmatch_lines(["*--list-overrides*"])

    def test_override_applied_through_cli(self, pytester):
        """A test file calling flag_gems.abs() sees the overridden impl."""
        pytester.makeini(f"""
            [pytest]
            pythonpath = {self._repo_src_path()}
            """)
        pytester.makeconftest("""
            from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

            def pytest_addoption(parser):
                add_override_arguments(parser)

            def pytest_configure(config):
                config._override_registry = apply_overrides_from_args(config.option)

            def pytest_unconfigure(config):
                if hasattr(config, "_override_registry"):
                    config._override_registry.restore_all()
            """)

        custom_impl = pytester.makepyfile(custom_abs="""
            import torch

            def my_abs(input):
                result = torch.abs(input)
                result._custom_marker = "from_cli_override"
                return result
            """)

        pytester.makepyfile(test_uses_override="""
            import torch
            import flag_gems

            def test_abs_is_overridden():
                x = torch.tensor([-1.0, -2.0], device=flag_gems.device)
                result = flag_gems.abs(x)
                assert getattr(result, "_custom_marker", None) == "from_cli_override"
            """)

        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            f"--override=abs:{custom_impl}:my_abs",
            "test_uses_override.py",
        )
        result.assert_outcomes(passed=1)

    def test_override_config_file_applied_through_cli(self, pytester):
        """--override-config loads a YAML file and applies the overrides."""
        pytester.makeini(f"""
            [pytest]
            pythonpath = {self._repo_src_path()}
            """)
        pytester.makeconftest("""
            from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

            def pytest_addoption(parser):
                add_override_arguments(parser)

            def pytest_configure(config):
                config._override_registry = apply_overrides_from_args(config.option)

            def pytest_unconfigure(config):
                if hasattr(config, "_override_registry"):
                    config._override_registry.restore_all()
            """)

        custom_impl = pytester.makepyfile(custom_neg="""
            import torch

            def my_neg(input):
                result = torch.neg(input)
                result._custom_marker = "from_config_file"
                return result
            """)

        config_file = pytester.makefile(
            ".yaml",
            overrides=f"""
            overrides:
              neg:
                file: {custom_impl}
                function: my_neg
            """,
        )

        pytester.makepyfile(test_uses_config_override="""
            import torch
            import flag_gems

            def test_neg_is_overridden():
                x = torch.tensor([1.0, -2.0], device=flag_gems.device)
                result = flag_gems.neg(x)
                assert getattr(result, "_custom_marker", None) == "from_config_file"
            """)

        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            f"--override-config={config_file}",
            "test_uses_config_override.py",
        )
        result.assert_outcomes(passed=1)

    def test_no_override_leaves_default_implementation(self, pytester):
        """Without --override, flag_gems ops run unmodified."""
        pytester.makeini(f"""
            [pytest]
            pythonpath = {self._repo_src_path()}
            """)
        pytester.makeconftest("""
            from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

            def pytest_addoption(parser):
                add_override_arguments(parser)

            def pytest_configure(config):
                config._override_registry = apply_overrides_from_args(config.option)

            def pytest_unconfigure(config):
                if hasattr(config, "_override_registry"):
                    config._override_registry.restore_all()
            """)

        pytester.makepyfile(test_no_override="""
            import torch
            import flag_gems

            def test_abs_not_overridden():
                x = torch.tensor([-1.0, -2.0], device=flag_gems.device)
                result = flag_gems.abs(x)
                assert not hasattr(result, "_custom_marker")
            """)

        result = pytester.runpytest("-p", "no:cacheprovider", "test_no_override.py")
        result.assert_outcomes(passed=1)

    @pytest.mark.parametrize("configuration", ["cli", "yaml", "json"])
    def test_missing_operator_is_available_during_collection(
        self, pytester, monkeypatch, configuration
    ):
        monkeypatch.delattr(flag_gems, "sparse_csr_tensor", raising=False)
        pytester.makeini(f"[pytest]\npythonpath = {self._repo_src_path()}\n")
        pytester.makeconftest("""
            import flag_gems
            from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

            def pytest_addoption(parser):
                add_override_arguments(parser)

            def pytest_configure(config):
                config._override_registry = apply_overrides_from_args(config.option)

            def pytest_unconfigure(config):
                config._override_registry.restore_all()
                assert not hasattr(flag_gems, "sparse_csr_tensor")
            """)
        candidate = pytester.makepyfile(candidate="def run(value): return value + 3")
        pytester.makepyfile(test_candidate="""
            import flag_gems

            assert callable(flag_gems.sparse_csr_tensor)

            def test_direct_call():
                assert flag_gems.sparse_csr_tensor(4) == 7
            """)
        if configuration == "cli":
            option = f"--override=sparse_csr_tensor:{candidate}:run"
        else:
            import json

            # JSON is also valid YAML, so both suffixes exercise config dispatch.
            config = pytester.makefile(
                f".{configuration}",
                overrides=json.dumps(
                    {
                        "overrides": {
                            "sparse_csr_tensor": {
                                "file": str(candidate),
                                "function": "run",
                            }
                        }
                    }
                ),
            )
            option = f"--override-config={config}"
        result = pytester.runpytest(
            "-p", "no:cacheprovider", option, "test_candidate.py"
        )
        result.assert_outcomes(passed=1)
        assert result.ret == 0


def _load_submodule(monkeypatch, name, path):
    """Load `path` as `name` and register it in sys.modules (via monkeypatch)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_flag_gems(monkeypatch):
    """A minimal stand-in `flag_gems` package (no Torch/GPU import needed).

    `dynamic_registry` and `cli_override` are loaded as real submodules of
    this fake package, so they exercise the actual production code; only
    the top-level `flag_gems` package (and its heavy op imports) is stubbed
    out.
    """
    gems = types.ModuleType("flag_gems")
    gems.__path__ = [str(ROOT / "src/flag_gems")]
    gems.softmax = lambda x: ("original", x)
    monkeypatch.setitem(sys.modules, "flag_gems", gems)
    registry = _load_submodule(
        monkeypatch,
        "flag_gems.dynamic_registry",
        ROOT / "src/flag_gems/dynamic_registry.py",
    )
    cli = _load_submodule(
        monkeypatch, "flag_gems.cli_override", ROOT / "src/flag_gems/cli_override.py"
    )
    return gems, registry, cli


def _apply(cli, spec):
    return cli.apply_overrides_from_args(
        types.SimpleNamespace(override=[spec], override_config=None)
    )


class TestRegistrarLiveOverrideResolution:
    """
    Tests for `GeneralOpRegistrar._resolve_live_override`, which makes sure
    that a runtime override applied via `DynamicOpOverride` is picked up
    when the op is (re-)registered, instead of the stale reference captured
    in the config tuple at import time.

    These tests avoid importing the real `flag_gems` package (and thus a
    GPU/Torch dependency) by loading `dynamic_registry.py` and
    `cli_override.py` as submodules of a minimal fake `flag_gems` module,
    and by exec'ing just the `GeneralOpRegistrar` class body from
    `op_registrar.py` with host-only device/library stubs.
    """

    def test_pytest_parser_accepts_the_plugin(self, fake_flag_gems):
        from _pytest.config.argparsing import Parser

        fake_flag_gems[2].add_override_arguments(Parser())

    def test_missing_candidate_aborts(self, fake_flag_gems, tmp_path):
        with pytest.raises((Exception, SystemExit)):
            _apply(fake_flag_gems[2], f'softmax:{tmp_path / "absent.py"}:run')

    def test_missing_run_aborts(self, fake_flag_gems, tmp_path):
        path = tmp_path / "candidate.py"
        path.write_text("def another_function(x): return x\n")
        with pytest.raises((Exception, SystemExit)):
            _apply(fake_flag_gems[2], f"softmax:{path}:run")

    def test_unknown_unused_operator_fails_and_is_removed(
        self, fake_flag_gems, tmp_path
    ):
        gems, _, cli = fake_flag_gems
        path = tmp_path / "candidate.py"
        path.write_text("def run(x): return x\n")
        registry = _apply(cli, f"softamx:{path}:run")
        with pytest.raises(AssertionError, match="never invoked"):
            registry.restore_all()
        assert not hasattr(gems, "softamx")
        assert registry.list_overrides() == []

    def test_existing_registration_uses_candidate(self, fake_flag_gems):
        """Registering `_softmax` should pick up an active override for `softmax`."""
        gems, registry_module, _ = fake_flag_gems
        # Execute the real registrar class with host-only device/library stubs.
        tree = ast.parse((ROOT / "src/flag_gems/runtime/op_registrar.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "GeneralOpRegistrar"
        )
        scope = {
            "DeviceDetector": lambda: types.SimpleNamespace(
                dispatch_key="CUDA", vendor="test", vendor_name="test"
            ),
            "common": types.SimpleNamespace(
                vendors=types.SimpleNamespace(CAMBRICON="cambricon")
            ),
            "backend": types.SimpleNamespace(get_unused_ops=lambda vendor: []),
        }
        exec(
            compile(
                ast.Module(body=[cls], type_ignores=[]), "<registrar class>", "exec"
            ),
            scope,
        )
        config = (("_softmax", gems.softmax),)
        registered = {}
        library = types.SimpleNamespace(
            impl=lambda key, fn, device: registered.update({key: fn})
        )
        candidate = lambda x: ("candidate", x)
        with registry_module.DynamicOpOverride() as registry:
            assert registry.override("softmax", candidate)
            assert gems.softmax is candidate
            scope["GeneralOpRegistrar"](config, lib=library)
            assert registered["_softmax"] is candidate

    @pytest.mark.parametrize("operator", ["softmax", "sparse_csr_tensor", "softamx"])
    def test_unused_candidate_makes_pytest_fail(self, tmp_path, operator):
        """A --override candidate that is never invoked should fail the run."""
        candidate = tmp_path / "candidate.py"
        candidate.write_text(
            'def run(x): raise AssertionError("CANDIDATE WAS CALLED")\n'
        )
        (tmp_path / "conftest.py").write_text(f"""
import sys, types
pkg = types.ModuleType("flag_gems")
pkg.__path__ = [{str(ROOT / 'src/flag_gems')!r}]
pkg.softmax = lambda x: x
pkg.neg = lambda x: -x
sys.modules["flag_gems"] = pkg
from flag_gems.cli_override import apply_overrides_from_args

def pytest_configure(config):
    config._override_registry = apply_overrides_from_args(types.SimpleNamespace(
        override=[{f"{operator}:{candidate}:run"!r}], override_config=None))

def pytest_unconfigure(config):
    config._override_registry.restore_all()
""")
        (tmp_path / "test_other.py").write_text(
            "import flag_gems\ndef test_other(): assert flag_gems.neg(1) == -1\n"
        )
        env = os.environ.copy()
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env.pop("PYTEST_ADDOPTS", None)
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--confcutdir",
                str(tmp_path),
                str(tmp_path / "test_other.py"),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, result.stdout + result.stderr
        assert "never invoked" in result.stdout + result.stderr

    def test_direct_call_and_restore_work(self, fake_flag_gems):
        gems, registry_module, _ = fake_flag_gems
        original = gems.softmax
        candidate = lambda x: ("candidate", x)
        with registry_module.DynamicOpOverride() as registry:
            assert registry.override("softmax", candidate)
            assert gems.softmax(1) == ("candidate", 1)
        assert gems.softmax is original


class TestUnimplementedOperatorOverride:
    @pytest.mark.parametrize("original_kind", ["missing", "none", "callable"])
    def test_repeated_override_restores_original_state(
        self, fake_flag_gems, original_kind
    ):
        gems, registry_module, _ = fake_flag_gems
        original = (
            (lambda value: ("original", value)) if original_kind == "callable" else None
        )
        if original_kind != "missing":
            gems.sparse_csr_tensor = original

        with registry_module.DynamicOpOverride() as registry:
            assert registry.override(
                "sparse_csr_tensor", lambda value: ("first", value)
            )
            assert gems.sparse_csr_tensor(1) == ("first", 1)
            assert registry.override(
                "sparse_csr_tensor", lambda value: ("second", value)
            )
            assert gems.sparse_csr_tensor(2) == ("second", 2)

        if original_kind == "missing":
            assert not hasattr(gems, "sparse_csr_tensor")
        else:
            assert gems.sparse_csr_tensor is original
        assert registry.list_overrides() == []

    def test_exception_restores_existing_and_removes_added_operator(
        self, fake_flag_gems
    ):
        gems, registry_module, _ = fake_flag_gems
        original = gems.softmax
        with pytest.raises(RuntimeError, match="test failed"):
            with registry_module.DynamicOpOverride() as registry:
                assert all(
                    registry.override_batch(
                        {
                            "softmax": lambda value: value + 1,
                            "sparse_csr_tensor": lambda value: value + 2,
                        }
                    ).values()
                )
                assert gems.softmax(1) == 2
                assert gems.sparse_csr_tensor(1) == 3
                raise RuntimeError("test failed")
        assert gems.softmax is original
        assert not hasattr(gems, "sparse_csr_tensor")

    @pytest.mark.parametrize("configuration", ["cli", "yaml", "json"])
    def test_file_candidate_for_missing_operator(
        self, fake_flag_gems, tmp_path, configuration
    ):
        gems, _, cli = fake_flag_gems
        candidate = tmp_path / "candidate.py"
        candidate.write_text("def run(value): return value + 3\n")
        args = types.SimpleNamespace(override=None, override_config=None)
        if configuration == "cli":
            args.override = [f"sparse_csr_tensor:{candidate}:run"]
        else:
            import json

            config = tmp_path / f"overrides.{configuration}"
            payload = {
                "overrides": {
                    "sparse_csr_tensor": {"file": str(candidate), "function": "run"}
                }
            }
            if configuration == "yaml":
                import yaml

                config.write_text(yaml.safe_dump(payload))
            else:
                config.write_text(json.dumps(payload))
            args.override_config = str(config)

        with cli.apply_overrides_from_args(args) as registry:
            assert gems.sparse_csr_tensor(4) == 7
            assert registry.list_overrides() == ["flag_gems.sparse_csr_tensor"]
        assert not hasattr(gems, "sparse_csr_tensor")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
