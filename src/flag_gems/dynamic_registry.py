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
Dynamic operator override interface for flexible testing.

This module provides utilities to dynamically override operator implementations
in the flag_gems module without modifying source code, enabling:
- Concurrent testing of different operator implementations
- Command-line driven operator override
- A/B testing of operator variants
"""

import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


class DynamicOpOverride:
    """
    Manage dynamic operator implementation override in flag_gems module.

    This class allows runtime override of operator implementations that are
    directly called via flag_gems.op_name() API, without modifying source code.

    Example:
        >>> registry = DynamicOpOverride()
        >>> # Override an implementation
        >>> registry.override("softmax", my_custom_softmax)
        >>> # Test it
        >>> result = flag_gems.softmax(x)
        >>> # Restore original
        >>> registry.restore("softmax")
    """

    def __init__(self):
        self._overrides: Dict[str, Any] = {}
        self._originals: Dict[str, Any] = {}

    def override(
        self,
        op_name: str,
        impl_func: Callable,
        module_name: str = "flag_gems",
    ) -> bool:
        """
        Override an operator implementation in the specified module.

        Args:
            op_name: Operator name (e.g., "softmax", "rms_norm", "linalg_matrix_exp")
            impl_func: New implementation function
            module_name: Module name to override in (default: "flag_gems")

        Returns:
            True if override succeeded, False otherwise
        """
        try:
            # Import the module
            if module_name not in sys.modules:
                __import__(module_name)
            module = sys.modules[module_name]

            # Check if the operator exists
            if not hasattr(module, op_name):
                warnings.warn(
                    f"Operator '{op_name}' not found in module '{module_name}'. "
                    f"Override will be applied but may not be effective."
                )

            # Store original implementation if not already stored
            full_name = f"{module_name}.{op_name}"
            if full_name not in self._originals:
                if hasattr(module, op_name):
                    self._originals[full_name] = getattr(module, op_name)
                else:
                    self._originals[full_name] = None

            # Override the implementation
            setattr(module, op_name, impl_func)
            self._overrides[full_name] = impl_func

            return True

        except Exception as e:
            warnings.warn(f"Failed to override '{op_name}': {e}")
            return False

    def override_batch(
        self,
        op_mapping: Dict[str, Callable],
        module_name: str = "flag_gems",
    ) -> Dict[str, bool]:
        """
        Override multiple operators at once.

        Args:
            op_mapping: Dict mapping op_name to implementation function
            module_name: Module name to override in (default: "flag_gems")

        Returns:
            Dict mapping op_name to override success status
        """
        results = {}
        for op_name, impl_func in op_mapping.items():
            results[op_name] = self.override(op_name, impl_func, module_name)
        return results

    def restore(
        self,
        op_name: str,
        module_name: str = "flag_gems",
    ) -> bool:
        """
        Restore an operator to its original implementation.

        Args:
            op_name: Operator name to restore
            module_name: Module name (default: "flag_gems")

        Returns:
            True if restoration succeeded, False otherwise
        """
        full_name = f"{module_name}.{op_name}"

        if full_name not in self._originals:
            warnings.warn(
                f"Operator '{op_name}' was not overridden by this registry"
            )
            return False

        try:
            if module_name not in sys.modules:
                warnings.warn(f"Module '{module_name}' not loaded")
                return False

            module = sys.modules[module_name]
            original = self._originals[full_name]

            if original is not None:
                setattr(module, op_name, original)
            elif hasattr(module, op_name):
                delattr(module, op_name)

            # Clean up tracking
            del self._originals[full_name]
            if full_name in self._overrides:
                del self._overrides[full_name]

            return True

        except Exception as e:
            warnings.warn(f"Failed to restore '{op_name}': {e}")
            return False

    def restore_all(self, module_name: str = "flag_gems"):
        """
        Restore all overridden operators in the specified module.

        Args:
            module_name: Module name (default: "flag_gems")
        """
        # Get list of ops to restore for this module
        ops_to_restore = [
            full_name
            for full_name in list(self._originals.keys())
            if full_name.startswith(f"{module_name}.")
        ]

        for full_name in ops_to_restore:
            op_name = full_name.split(".", 1)[1]
            self.restore(op_name, module_name)

    def list_overrides(self) -> List[str]:
        """Return list of currently overridden operator names."""
        return list(self._overrides.keys())

    def load_impl_from_file(self, filepath: str, func_name: str) -> Optional[Callable]:
        """
        Load an implementation function from a Python file.

        Args:
            filepath: Path to Python file containing the implementation
            func_name: Name of the function to load

        Returns:
            The loaded function, or None if loading failed
        """
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")

            spec = importlib.util.spec_from_file_location(f"custom_impl_{path.stem}", filepath)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load spec from {filepath}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, func_name):
                raise AttributeError(f"Function '{func_name}' not found in {filepath}")

            return getattr(module, func_name)

        except Exception as e:
            warnings.warn(f"Failed to load implementation from {filepath}: {e}")
            return None

    def override_from_file(
        self,
        op_name: str,
        filepath: str,
        func_name: Optional[str] = None,
        module_name: str = "flag_gems",
    ) -> bool:
        """
        Override an operator implementation from a Python file.

        Args:
            op_name: Operator name to override
            filepath: Path to Python file containing the implementation
            func_name: Function name to load (defaults to op_name)
            module_name: Module name to override in (default: "flag_gems")

        Returns:
            True if override succeeded, False otherwise
        """
        if func_name is None:
            func_name = op_name

        impl_func = self.load_impl_from_file(filepath, func_name)
        if impl_func is None:
            return False

        return self.override(op_name, impl_func, module_name)

    def override_batch_from_files(
        self,
        op_file_mapping: Dict[str, Union[str, tuple]],
        module_name: str = "flag_gems",
    ) -> Dict[str, bool]:
        """
        Override multiple operators from files.

        Args:
            op_file_mapping: Dict mapping op_name to either:
                - A string filepath (function name same as op_name)
                - A tuple (filepath, function_name)
            module_name: Module name to override in (default: "flag_gems")

        Returns:
            Dict mapping op_name to override success status
        """
        results = {}
        for op_name, file_spec in op_file_mapping.items():
            if isinstance(file_spec, str):
                filepath = file_spec
                func_name = op_name
            elif isinstance(file_spec, (tuple, list)) and len(file_spec) == 2:
                filepath, func_name = file_spec
            else:
                warnings.warn(f"Invalid file spec for '{op_name}': {file_spec}")
                results[op_name] = False
                continue

            results[op_name] = self.override_from_file(
                op_name, filepath, func_name, module_name
            )

        return results

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore all overrides."""
        self.restore_all()

    def __del__(self):
        """Destructor - restore on deletion."""
        self.restore_all()


# Global registry instance for convenience
_global_registry = DynamicOpOverride()


def override_op(
    op_name: str,
    impl_func: Callable,
    module_name: str = "flag_gems",
) -> bool:
    """
    Convenience function to override a single operator using global registry.

    Args:
        op_name: Operator name
        impl_func: New implementation function
        module_name: Module name (default: "flag_gems")

    Returns:
        True if override succeeded
    """
    return _global_registry.override(op_name, impl_func, module_name)


def restore_op(op_name: str, module_name: str = "flag_gems") -> bool:
    """
    Convenience function to restore a single operator using global registry.

    Args:
        op_name: Operator name
        module_name: Module name (default: "flag_gems")

    Returns:
        True if restoration succeeded
    """
    return _global_registry.restore(op_name, module_name)


def restore_all_ops(module_name: str = "flag_gems"):
    """
    Convenience function to restore all operators using global registry.

    Args:
        module_name: Module name (default: "flag_gems")
    """
    _global_registry.restore_all(module_name)


def get_global_registry() -> DynamicOpOverride:
    """Get the global registry instance."""
    return _global_registry
