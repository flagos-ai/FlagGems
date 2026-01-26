import os
import warnings
from pathlib import Path

import yaml

# Optional imports used inside helper functions to avoid hard dependencies at
# module import time.
try:  # pragma: no cover - best effort fallback
    from flag_gems import runtime as _runtime
except Exception:  # noqa: BLE001
    _runtime = None

has_c_extension = False
use_c_extension = False
aten_patch_list = []

# set FLAGGEMS_SOURCE_DIR for cpp extension to find
os.environ["FLAGGEMS_SOURCE_DIR"] = str(Path(__file__).parent.resolve())

try:
    from flag_gems import c_operators

    has_c_extension = True
except ImportError:
    c_operators = None
    has_c_extension = False


use_env_c_extension = os.environ.get("USE_C_EXTENSION", "0") == "1"
if use_env_c_extension and not has_c_extension:
    warnings.warn(
        "[FlagGems] USE_C_EXTENSION is set, but C extension is not available. "
        "Falling back to pure Python implementation.",
        RuntimeWarning,
    )

if has_c_extension and use_env_c_extension:
    try:
        from flag_gems import aten_patch

        aten_patch_list = aten_patch.get_registered_ops()
        use_c_extension = True
    except (ImportError, AttributeError):
        aten_patch_list = []
        use_c_extension = False


def load_enable_config_from_yaml(yaml_path, key="include"):
    """
    Load include/exclude operator lists from a YAML file.

    Expected YAML structure:
        include:  # operators to explicitly enable
          - op_a
          - op_b
        exclude:  # operators to skip
          - op_c

    Both keys are optional; missing keys default to empty lists.
    Returns two lists `include` and `exclude`.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.is_file():
        warnings.warn(f"load_enable_config_from_yaml: yaml not found: {yaml_path}")
        return []

    try:
        data = yaml.safe_load(yaml_path.read_text())
    except Exception as err:
        warnings.warn(
            f"load_enable_config_from_yaml: unexpected error reading {yaml_path}: {err}"
        )
        return []

    if key not in ("include", "exclude"):
        warnings.warn(
            f"load_enable_config_from_yaml: key must be 'include' or 'exclude', got: {key}"
        )
        return []

    if data is None:
        return []

    if isinstance(data, dict):
        operator_list = list(set(data.get(key, [])))
        return operator_list

    warnings.warn(
        f"load_enable_config_from_yaml: yaml {yaml_path} must be a mapping with 'include'/'exclude' lists"
    )
    return []


def get_default_enable_config(vendor_name=None, arch_name=None):
    base_dir = Path(__file__).resolve().parent / "runtime" / "backend"
    vendor_dir = base_dir / f"_{vendor_name}" if vendor_name else base_dir

    candidates = []
    if vendor_dir.is_dir():
        if arch_name:
            candidates.append(vendor_dir / arch_name / "enable_configs.yaml")
        candidates.append(vendor_dir / "enable_configs.yaml")
    candidates.append(
        base_dir / "_nvidia" / "enable_configs.yaml"
    )  # use nvidia as default
    return candidates


def resolve_user_setting(user_setting_info, user_setting_type="include"):
    """
    Resolve user setting for include/exclude operator lists.

    Args:
        user_setting_info: Can be a list/tuple/set of operators, "default", None, or a path to a YAML file.
        user_setting_type: Either "include" or "exclude".

    Returns:
        List of operators based on the user setting.
    """
    # If user_setting_info is a list, tuple, or set, use it directly as the operator list (deduplicated)
    if isinstance(user_setting_info, (list, tuple, set)):
        return list(set(user_setting_info))

    yaml_candidates = []
    # If set to "default" or None (for include type),
    # load from default YAML config files based on vendor and architecture
    if user_setting_info == "default" or (
        user_setting_type == "include" and user_setting_info is None
    ):
        # Lazily infer vendor/arch if not provided.
        vendor_name = _runtime.device.vendor_name
        arch_event = _runtime.backend.BackendArchEvent()
        if arch_event.has_arch:
            arch_name = getattr(arch_event, "arch", None)
        yaml_candidates = get_default_enable_config(vendor_name, arch_name)

    # If user_setting_info is a string, treat it as a YAML file path
    elif isinstance(user_setting_info, str):
        yaml_candidates.append(user_setting_info)

    # Iterate through candidate YAML paths and try to load the operator list
    for yaml_path in yaml_candidates:
        operator_list = load_enable_config_from_yaml(yaml_path, user_setting_type)
        if operator_list:
            return operator_list
        else:
            warnings.warn(
                f"resolve_user_setting: {user_setting_type} yaml not found: {yaml_path}"
            )

    # If no operators found in any YAML, warn and return empty list
    warnings.warn(
        f"resolve_user_setting: no {user_setting_type} ops found; returning empty list"
    )
    return []


__all__ = [
    "aten_patch_list",
    "has_c_extension",
    "use_c_extension",
    "resolve_user_setting",
]
