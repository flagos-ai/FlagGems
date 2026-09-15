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
"""

import tempfile
from pathlib import Path

import pytest
import torch

import flag_gems
from flag_gems.dynamic_registry import DynamicOpOverride

# Needed for TestPytestIntegration, which uses the `pytester` fixture to
# run pytest itself as a subprocess and verify the --override/--override-config
# CLI options work end-to-end through the real pytest hooks.
pytest_plugins = ["pytester"]


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

        assert hasattr(result, "_custom_marker"), "Custom implementation should be called"
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
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
            success = registry.override_from_file(
                "abs",
                temp_file,
                "my_custom_abs"
            )
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
            results = registry.override_batch_from_files({
                "abs": (str(abs_file), "my_abs"),
                "neg": (str(neg_file), "my_neg"),
            })

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
            "abs",
            "/nonexistent/path/file.py",
            "some_func"
        )

        assert not success, "Should fail for nonexistent file"

    def test_override_missing_function(self):
        """Test handling of missing function in file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def some_other_function():
    pass
""")
            temp_file = f.name

        try:
            registry = DynamicOpOverride()

            success = registry.override_from_file(
                "abs",
                temp_file,
                "nonexistent_function"
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
        registry2 = DynamicOpOverride()

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
        op_name, filepath, func_name = parse_override_spec("softmax:./custom.py:my_softmax")
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
        op_name, filepath, func_name = parse_override_spec("softmax=./custom.py:my_softmax")
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

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
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

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
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
        pytester.makeini(
            f"""
            [pytest]
            pythonpath = {self._repo_src_path()}
            """
        )
        pytester.makeconftest(
            """
            from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

            def pytest_addoption(parser):
                add_override_arguments(parser)

            def pytest_configure(config):
                config._override_registry = apply_overrides_from_args(config.option)

            def pytest_unconfigure(config):
                if hasattr(config, "_override_registry"):
                    config._override_registry.restore_all()
            """
        )

        result = pytester.runpytest("--help")
        result.stdout.fnmatch_lines(["*--override=SPEC*"])
        result.stdout.fnmatch_lines(["*--override-config=PATH*"])
        result.stdout.fnmatch_lines(["*--list-overrides*"])

    def test_override_applied_through_cli(self, pytester):
        """A test file calling flag_gems.abs() sees the overridden impl."""
        pytester.makeini(
            f"""
            [pytest]
            pythonpath = {self._repo_src_path()}
            """
        )
        pytester.makeconftest(
            """
            from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

            def pytest_addoption(parser):
                add_override_arguments(parser)

            def pytest_configure(config):
                config._override_registry = apply_overrides_from_args(config.option)

            def pytest_unconfigure(config):
                if hasattr(config, "_override_registry"):
                    config._override_registry.restore_all()
            """
        )

        custom_impl = pytester.makepyfile(
            custom_abs="""
            import torch

            def my_abs(input):
                result = torch.abs(input)
                result._custom_marker = "from_cli_override"
                return result
            """
        )

        pytester.makepyfile(
            test_uses_override="""
            import torch
            import flag_gems

            def test_abs_is_overridden():
                x = torch.tensor([-1.0, -2.0], device=flag_gems.device)
                result = flag_gems.abs(x)
                assert getattr(result, "_custom_marker", None) == "from_cli_override"
            """
        )

        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            f"--override=abs:{custom_impl}:my_abs",
            "test_uses_override.py",
        )
        result.assert_outcomes(passed=1)

    def test_override_config_file_applied_through_cli(self, pytester):
        """--override-config loads a YAML file and applies the overrides."""
        pytester.makeini(
            f"""
            [pytest]
            pythonpath = {self._repo_src_path()}
            """
        )
        pytester.makeconftest(
            """
            from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

            def pytest_addoption(parser):
                add_override_arguments(parser)

            def pytest_configure(config):
                config._override_registry = apply_overrides_from_args(config.option)

            def pytest_unconfigure(config):
                if hasattr(config, "_override_registry"):
                    config._override_registry.restore_all()
            """
        )

        custom_impl = pytester.makepyfile(
            custom_neg="""
            import torch

            def my_neg(input):
                result = torch.neg(input)
                result._custom_marker = "from_config_file"
                return result
            """
        )

        config_file = pytester.makefile(
            ".yaml",
            overrides=f"""
            overrides:
              neg:
                file: {custom_impl}
                function: my_neg
            """,
        )

        pytester.makepyfile(
            test_uses_config_override="""
            import torch
            import flag_gems

            def test_neg_is_overridden():
                x = torch.tensor([1.0, -2.0], device=flag_gems.device)
                result = flag_gems.neg(x)
                assert getattr(result, "_custom_marker", None) == "from_config_file"
            """
        )

        result = pytester.runpytest(
            "-p",
            "no:cacheprovider",
            f"--override-config={config_file}",
            "test_uses_config_override.py",
        )
        result.assert_outcomes(passed=1)

    def test_no_override_leaves_default_implementation(self, pytester):
        """Without --override, flag_gems ops run unmodified."""
        pytester.makeini(
            f"""
            [pytest]
            pythonpath = {self._repo_src_path()}
            """
        )
        pytester.makeconftest(
            """
            from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

            def pytest_addoption(parser):
                add_override_arguments(parser)

            def pytest_configure(config):
                config._override_registry = apply_overrides_from_args(config.option)

            def pytest_unconfigure(config):
                if hasattr(config, "_override_registry"):
                    config._override_registry.restore_all()
            """
        )

        pytester.makepyfile(
            test_no_override="""
            import torch
            import flag_gems

            def test_abs_not_overridden():
                x = torch.tensor([-1.0, -2.0], device=flag_gems.device)
                result = flag_gems.abs(x)
                assert not hasattr(result, "_custom_marker")
            """
        )

        result = pytester.runpytest(
            "-p", "no:cacheprovider", "test_no_override.py"
        )
        result.assert_outcomes(passed=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
