import ast
import functools
import importlib
import inspect
import os
import sys
from pathlib import Path

from ..common import vendors
from . import backend_utils

class BackendState:
    """Singleton class to manage backend state variables."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.vendor_module = None
        self.device_name = None
        self.torch_device_object = None
        self.torch_device_fn_device = None
        self.tl_extra_backend_module = None
        self.ops_module = None
        self.fused_module = None
        self.heuristic_config_module = None
        self.vendor_extra_lib_imported = False
        self.device_fn_cache = {}
        self.customized_ops = None


# Global singleton instance 
_state = BackendState()


class BackendArchEvent:
    has_arch: bool = False
    _instance = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, backend=None):
        if BackendArchEvent._initialized:
            return
        BackendArchEvent._initialized = True
        self.backend = backend
        self.error_msgs = []
        self.arch = self.get_arch()
        if self.has_arch:
            self.supported_archs = self._get_supported_archs()
            # current_arch_path is like FlagGems/src/flag_gems/runtime/backend/_nvidia/hopper
            self.current_arch_path = self.supported_archs.get(self.arch)
            self.arch_module = self.get_arch_module()
            self.autotune_configs = self.get_autotune_configs()
            self.heuristics_configs = self.get_heuristics_configs()

    def get_functions_from_module(self, module):
        return inspect.getmembers(module, inspect.isfunction) if module else []

    def get_heuristics_configs(self):
        try:
            heuristic_module = self.arch_module
        except Exception:  # noqa E722
            sys.path.insert(0, str(self.current_arch_path))
            heuristic_module = importlib.import_module("heuristics_config_utils")
            sys.path.remove(str(self.current_arch_path))
        return getattr(heuristic_module, "HEURISTICS_CONFIGS", None)

    def get_autotune_configs(self):
        path = self.current_arch_path
        return backend_utils.get_tune_config(file_path=path)

    def get_arch(self, device=0):
        if not hasattr(_state.vendor_module, "ARCH_MAP"):
            return
        arch_map = _state.vendor_module.ARCH_MAP
        arch_string = os.environ.get("ARCH", "")
        arch_string_num = arch_string.split("_")[-1][0] if arch_string else arch_string
        if not arch_string_num:
            try:
                if not _state.torch_device_object.is_available():
                    return False
                props = _state.torch_device_object.get_device_properties(device)
                arch_string_num = str(props.major)
            except Exception:
                self.has_arch = False
        if arch_string_num not in arch_map:
            print(
                f"[INFO] : FlagGems Unsupported GPU arch {arch_string} specialization"
            )
        else:
            self.has_arch = True
            return arch_map[arch_string_num]

    def _get_supported_archs(self, path=None):
        path = Path(path or _state.vendor_module.__path__[0])
        path = path.parent if path.is_file() else path
        excluded = ("ops", "fused")
        return {
            p.name: str(p)
            for p in path.iterdir()
            if p.is_dir() and p.name not in excluded and not p.name.startswith("_")
        }

    def get_supported_archs(self):
        return list(self.supported_archs.keys())

    def get_arch_module(self):
        """Load backend.<arch>"""
        path_dir = os.path.dirname(self.current_arch_path)
        sys.path.insert(0, str(path_dir))
        current_arch_module = importlib.import_module(self.arch)
        sys.path.remove(str(path_dir))
        return current_arch_module

    def get_arch_ops(self):
        arch_specialized_ops = []
        sys.path.append(self.current_arch_path)
        ops_module = getattr(self.arch_module, "ops", None)
        try:
            if ops_module is None:
                ops_module = importlib.import_module(f"{self.arch}.ops")
        except Exception:
            try:
                sys.path.append(self.current_arch_path)
                ops_module = importlib.import_module(f"{self.arch}.ops")
                arch_specialized_ops.extend(self.get_functions_from_module(ops_module))
            except Exception as err_msg:
                self.error_msgs.append(err_msg)

        return arch_specialized_ops


def _import_module_safe(module_name, vendor_name, module_type):
    """Helper to import a module with proper error handling."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        print(
           f"[Note] No specialized {module_type} operators were found for "
           f"the {vendor_name}, generic {module_type} operators will be used by default."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to import vendor extra lib: {e}")



def import_vendor_extra_lib(vendor_name=None):
    if _state.vendor_extra_lib_imported:
        return
    _state.ops_module = _import_module_safe(
        f"_{vendor_name}.ops", vendor_name, "common"
    )
    _state.fused_module = _import_module_safe(
        f"_{vendor_name}.fused", vendor_name, "fused"
    )
    _state.vendor_extra_lib_imported = True


def get_codegen_result(code, result_key):
    parsed_ast = ast.parse(code)
    compiled_code = compile(parsed_ast, filename="<ast>", mode="exec")
    try:
        exec(compiled_code, globals())
    except Exception as e:
        raise e
    return globals()[result_key]


@functools.lru_cache(maxsize=32)
def gen_torch_tensor_attr_res(tensor, attr_name):
    _state.device_name = _state.device_name or get_vendor_info().device_name
    code = f"""
import torch
res = {tensor}.{attr_name}
    """
    return get_codegen_result(code, "res")


def set_tl_extra_backend_module(vendor_name=None):
    vendor_info = get_vendor_info(vendor_name)
    _state.device_name = _state.device_name or vendor_info.device_name
    extra_name = vendor_info.triton_extra_name or _state.device_name
    module_str = f"triton.language.extra.{extra_name}.libdevice"
    _state.tl_extra_backend_module = importlib.import_module(module_str)


def get_tl_extra_backend_module():
    return _state.tl_extra_backend_module


def set_torch_backend_device_fn(vendor_name=None):
    _state.device_name = _state.device_name or get_vendor_info(vendor_name).device_name
    module_str = f"torch.backends.{_state.device_name}"
    if _state.device_name in ("musa", "aipu", "npu", "txda", "ptpu", "gcu"):
        _state.torch_device_fn_device = None
    else:
        _state.torch_device_fn_device = importlib.import_module(module_str)


def get_torch_backend_device_fn():
    return _state.torch_device_fn_device


def gen_torch_device_object(vendor_name=None):
    if _state.torch_device_object is not None:
        return _state.torch_device_object
    _state.device_name = _state.device_name or get_vendor_info(vendor_name).device_name
    code = f"""
import torch
fn = torch.{_state.device_name}
"""
    _state.torch_device_object = get_codegen_result(code, "fn")
    return _state.torch_device_object


def get_vendor_module(vendor_name, query=False):
    def get_module(vendor_name):
        current_file_path = os.path.abspath(__file__)
        current_dir_path = os.path.dirname(current_file_path)
        sys.path.append(current_dir_path)
        return importlib.import_module(vendor_name)

    if (
        query
    ):  # The purpose of a query is to provide the user with the instance that he wants to import
        return get_module(vendor_name)

    if _state.vendor_module is None:
        _state.vendor_module = get_module("_" + vendor_name) 
    return _state.vendor_module


def get_vendor_info(vendor_name=None, query=False):
    if query:
        return get_vendor_module(vendor_name, query).vendor_info
    global vendor_module  # noqa: F824
    get_vendor_module(vendor_name)
    return vendor_module.vendor_info


def get_vendor_infos():
    infos = []
    for vendor_name in vendors.get_all_vendors():
        try:
            infos.append(get_vendor_info(f"_{vendor_name}", query=True))
        except Exception:
            continue

    return infos


def get_customized_ops(vendor_name=None):
    import_vendor_extra_lib(vendor_name)
    if _state.customized_ops is not None:
        return _state.customized_ops
    _state.customized_ops = []
    if _state.ops_module is not None:
        ops = inspect.getmembers(_state.ops_module, inspect.isfunction)
        _state.customized_ops += ops
    if _state.fused_module is not None:
        fused_ops = inspect.getmembers(_state.fused_module, inspect.isfunction)
        _state.customized_ops += fused_ops
    return _state.customized_ops


def get_unused_ops(vendor_name=None):
    global vendor_module  # noqa: F824
    get_vendor_module(vendor_name)
    return list(_state.vendor_module.CUSTOMIZED_UNUSED_OPS)


def get_heuristic_config(vendor_name=None):
    config_name = "heuristics_config_utils"
    default_backend = "nvidia"
    for backend in (vendor_name, default_backend):
        mod_name = f"_{backend}.{config_name}"
        try:
            _state.heuristic_config_module = importlib.import_module(mod_name)
        except Exception:
            continue 
    return getattr(_state.heuristic_config_module, "HEURISTICS_CONFIGS", None)


def get_tune_config(vendor_name=None):
    global vendor_module  # noqa: F824
    get_vendor_module(vendor_name)
    return backend_utils.get_tune_config(vendor_name)


def get_expand_config(op_name=None, file_path=None):
    return backend_utils.get_expand_config(op_name=op_name, file_path=file_path)

def get_backend_state() -> BackendState:
    """Get the global BackendState singleton instance."""
    return _state

__all__ = ["*"]
