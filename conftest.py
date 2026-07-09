import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC_PATH = _ROOT / "src"
_SRC = str(_SRC_PATH)

# Avoid importing another editable FlagGems checkout before the local src tree.
sys.meta_path = [
    finder
    for finder in sys.meta_path
    if finder.__class__.__module__ != "_flag_gems_editable"
]

if _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)

for name, module in list(sys.modules.items()):
    if name != "flag_gems" and not name.startswith("flag_gems."):
        continue
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        del sys.modules[name]
        continue
    try:
        Path(module_file).resolve().relative_to(_SRC_PATH)
    except ValueError:
        del sys.modules[name]
