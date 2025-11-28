import os
import warnings
from pathlib import Path

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
