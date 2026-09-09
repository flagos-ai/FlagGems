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
Command-line interface for dynamic operator override.

This module provides CLI utilities to override operator implementations
from command-line arguments, enabling flexible testing workflows.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .dynamic_registry import DynamicOpOverride


def parse_override_spec(spec: str) -> tuple:
    """
    Parse an override specification string.

    Formats:
        - "op_name:filepath" - load function with same name as op
        - "op_name:filepath:func_name" - load specified function
        - "op_name=filepath" - alternative syntax
        - "op_name=filepath:func_name" - alternative syntax

    Args:
        spec: Override specification string

    Returns:
        Tuple of (op_name, filepath, func_name)
    """
    # Support both : and = as separators
    if "=" in spec:
        parts = spec.split("=", 1)
        op_name = parts[0].strip()
        rest = parts[1].strip()
    elif ":" in spec:
        parts = spec.split(":", 1)
        op_name = parts[0].strip()
        rest = parts[1].strip()
    else:
        raise ValueError(f"Invalid override spec format: {spec}")

    # Parse the rest for filepath and optional function name
    if ":" in rest:
        filepath, func_name = rest.split(":", 1)
        filepath = filepath.strip()
        func_name = func_name.strip()
    else:
        filepath = rest.strip()
        func_name = op_name

    return op_name, filepath, func_name


def load_override_config(config_path: str) -> Dict[str, tuple]:
    """
    Load override configuration from YAML or JSON file.

    File format (YAML):
        overrides:
          softmax:
            file: ./custom_softmax.py
            function: my_softmax  # optional, defaults to op name
          rms_norm:
            file: ./custom_rms_norm.py

    Or simple format:
        overrides:
          softmax: ./custom_softmax.py
          rms_norm: ./custom_rms_norm.py:my_rms

    Args:
        config_path: Path to YAML or JSON config file

    Returns:
        Dict mapping op_name to (filepath, func_name)
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config
    with open(path) as f:
        if path.suffix in [".yml", ".yaml"]:
            config = yaml.safe_load(f)
        elif path.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    # Parse overrides
    overrides = {}
    override_dict = config.get("overrides", {})

    for op_name, spec in override_dict.items():
        if isinstance(spec, str):
            # Simple string format: "filepath" or "filepath:func_name"
            if ":" in spec:
                filepath, func_name = spec.split(":", 1)
            else:
                filepath = spec
                func_name = op_name
        elif isinstance(spec, dict):
            # Dict format with 'file' and optional 'function'
            filepath = spec.get("file")
            func_name = spec.get("function", op_name)
            if not filepath:
                raise ValueError(f"Missing 'file' key for operator '{op_name}'")
        else:
            raise ValueError(f"Invalid override spec for '{op_name}': {spec}")

        overrides[op_name] = (filepath, func_name)

    return overrides


def apply_overrides_from_args(
    args: argparse.Namespace,
    registry: Optional[DynamicOpOverride] = None,
) -> DynamicOpOverride:
    """
    Apply operator overrides based on parsed command-line arguments.

    Args:
        args: Parsed arguments with 'override' and/or 'override_config' attributes
        registry: Optional registry instance (creates new one if None)

    Returns:
        DynamicOpOverride instance with overrides applied
    """
    if registry is None:
        registry = DynamicOpOverride()

    # Apply overrides from config file
    if hasattr(args, "override_config") and args.override_config:
        try:
            config_overrides = load_override_config(args.override_config)
            for op_name, (filepath, func_name) in config_overrides.items():
                success = registry.override_from_file(op_name, filepath, func_name)
                if success:
                    print(f"✓ Overridden {op_name} from {filepath}:{func_name}")
                else:
                    print(f"✗ Failed to override {op_name}", file=sys.stderr)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)

    # Apply overrides from command-line specs
    if hasattr(args, "override") and args.override:
        for spec in args.override:
            try:
                op_name, filepath, func_name = parse_override_spec(spec)
                success = registry.override_from_file(op_name, filepath, func_name)
                if success:
                    print(f"✓ Overridden {op_name} from {filepath}:{func_name}")
                else:
                    print(f"✗ Failed to override {op_name}", file=sys.stderr)
            except Exception as e:
                print(f"Error parsing override spec '{spec}': {e}", file=sys.stderr)
                sys.exit(1)

    return registry


def add_override_arguments(parser: argparse.ArgumentParser):
    """
    Add override-related arguments to an argument parser.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    override_group = parser.add_argument_group("operator override options")

    override_group.add_argument(
        "--override",
        "-o",
        action="append",
        metavar="SPEC",
        help=(
            "Override operator implementation. Format: 'op_name:filepath[:func_name]' "
            "or 'op_name=filepath[:func_name]'. Can be specified multiple times. "
            "Example: --override softmax:./my_softmax.py:custom_softmax"
        ),
    )

    override_group.add_argument(
        "--override-config",
        "-c",
        metavar="PATH",
        help=(
            "Load operator overrides from YAML/JSON config file. "
            "Example: --override-config ./overrides.yaml"
        ),
    )

    override_group.add_argument(
        "--list-overrides",
        action="store_true",
        help="List all active operator overrides and exit",
    )


def create_test_wrapper_script(
    test_script: str,
    output_path: Optional[str] = None,
) -> str:
    """
    Create a wrapper script that applies overrides before running tests.

    Args:
        test_script: Path to the original test script
        output_path: Optional output path for wrapper (default: test_script_wrapped.py)

    Returns:
        Path to the created wrapper script
    """
    if output_path is None:
        test_path = Path(test_script)
        output_path = str(test_path.parent / f"{test_path.stem}_wrapped.py")

    wrapper_code = f'''#!/usr/bin/env python3
"""Auto-generated wrapper for {test_script} with dynamic override support."""

import sys
import argparse
from pathlib import Path

# Add flag_gems to path if needed
import flag_gems
from flag_gems.cli_override import add_override_arguments, apply_overrides_from_args

def main():
    parser = argparse.ArgumentParser(
        description="Run {test_script} with optional operator overrides"
    )
    add_override_arguments(parser)

    # Parse known args (let test script handle the rest)
    args, remaining = parser.parse_known_args()

    # Apply overrides
    registry = apply_overrides_from_args(args)

    if args.list_overrides:
        overrides = registry.list_overrides()
        if overrides:
            print("Active overrides:")
            for override in overrides:
                print(f"  - {{override}}")
        else:
            print("No active overrides")
        return 0

    # Run the original test script
    try:
        # Update sys.argv for the test script
        sys.argv = ["{test_script}"] + remaining

        # Execute the test script
        with open("{test_script}") as f:
            code = compile(f.read(), "{test_script}", "exec")
            exec(code, {{"__name__": "__main__"}})
    finally:
        # Cleanup overrides
        registry.restore_all()

    return 0

if __name__ == "__main__":
    sys.exit(main())
'''

    with open(output_path, "w") as f:
        f.write(wrapper_code)

    # Make executable
    Path(output_path).chmod(0o755)

    return output_path


if __name__ == "__main__":
    # Simple CLI for creating wrapper scripts
    parser = argparse.ArgumentParser(
        description="Create test wrapper scripts with override support"
    )
    parser.add_argument("test_script", help="Original test script path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for wrapper script (default: auto-generated)",
    )

    args = parser.parse_args()
    output = create_test_wrapper_script(args.test_script, args.output)
    print(f"Created wrapper script: {output}")
