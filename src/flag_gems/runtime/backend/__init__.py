import ast
import functools
import importlib
import inspect
import os
import sys
from pathlib import Path

from ..commom_utils import vendors
from . import backend_utils

vendor_module = None
device_name = None
torch_device_object = None
torch_device_fn_device = None
tl_extra_backend_module = None
ops_module = None
fused_module = None
heuristic_config_module = None
vendor_extra_lib_imported = False
device_fn_cache = {}
customized_ops = None


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
        heuristic_module = None
        try:
            heuristic_module = self.arch_module
        except Exception:  # noqa E722
            sys.path.insert(0, str(self.current_arch_path))
            heuristic_module = importlib.import_module("heuristics_config_utils")
            sys.path.remove(str(self.current_arch_path))
        if hasattr(heuristic_module, "HEURISTICS_CONFIGS"):
            return heuristic_module.HEURISTICS_CONFIGS
        return None

    def get_autotune_configs(self):
        path = self.current_arch_path
        return backend_utils.get_tune_config(file_path=path)

    def get_arch(self, device=0):
        if not hasattr(vendor_module, "ARCH_MAP"):
            return
        arch_map = vendor_module.ARCH_MAP
        arch_string = os.environ.get("ARCH", "")
        arch_string_num = arch_string.split("_")[-1][0] if arch_string else arch_string
        if not arch_string_num:
            try:
                if not torch_device_object.is_available():
                    return False
                props = torch_device_object.get_device_properties(device)
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
        path = path or vendor_module.__path__[0]
        excluded = ("ops", "fused")
        path = Path(path)
        path = path.parent if path.is_file() else path
        archs = {}
        for p in path.iterdir():
            name = str(p).split("/")[-1]
            if p.is_dir() and name not in excluded and not name.startswith("_"):
                archs.update({name: str(p)})
        return archs

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
        modules = []
        sys.path.append(self.current_arch_path)
        ops_module = importlib.import_module(f"{self.arch}.ops")
        try:
            ops_module = self.arch_module.ops
            modules.append(ops_module)
        except Exception:
            try:
                sys.path.append(self.current_arch_path)
                ops_module = importlib.import_module(f"{self.arch}.ops")
                modules.append(ops_module)
            except Exception as err_msg:
                self.error_msgs.append(err_msg)

        for mod in modules:
            arch_specialized_ops.extend(self.get_functions_from_module(mod))

        return arch_specialized_ops


def import_vendor_extra_lib(vendor_name=None):
    global vendor_extra_lib_imported
    if vendor_extra_lib_imported is True:
        return
    global ops_module, fused_module
    try:
        ops_module = importlib.import_module(f"_{vendor_name}.ops")
    except ModuleNotFoundError:
        print(
            f"[Note] No specialized common operators were found in"
            f"the {vendor_name} implementation, and general common operators are used by default."
        )
    except Exception as e:
        raise RuntimeError(f"Import vendor extra lib failed: {e}")

    try:
        fused_module = importlib.import_module(f"_{vendor_name}.fused")
    except ModuleNotFoundError:
        print(
            f"[Note] No specialized fused operators were found in"
            f"the {vendor_name} implementation, and general fused operators are used by default."
        )
    except Exception as e:
        raise RuntimeError(f"Import vendor extra lib failed: {e}")
    vendor_extra_lib_imported = True


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
    global device_name
    device_name = device_name or get_vendor_info().device_name
    code = f"""
import torch
res = {tensor}.{attr_name}
    """
    return get_codegen_result(code, "res")


def set_tl_extra_backend_module(vendor_name=None):
    global device_name, tl_extra_backend_module
    vendor_info = get_vendor_info(vendor_name)
    device_name = device_name or vendor_info.device_name
    extra_name = vendor_info.triton_extra_name or device_name
    module_str = f"triton.language.extra.{extra_name}.libdevice"
    tl_extra_backend_module = importlib.import_module(module_str)


def get_tl_extra_backend_module():
    return tl_extra_backend_module


def set_torch_backend_device_fn(vendor_name=None):
    global device_name, torch_device_fn_device
    device_name = device_name or get_vendor_info(vendor_name).device_name
    module_str = f"torch.backends.{device_name}"
    if device_name in ("musa", "aipu", "npu"):
        torch_device_fn_device = None
    else:
        torch_device_fn_device = importlib.import_module(module_str)


def get_torch_backend_device_fn():
    return torch_device_fn_device


def gen_torch_device_object(vendor_name=None):
    global device_name, torch_device_object
    if torch_device_object is not None:
        return torch_device_object
    device_name = device_name or get_vendor_info(vendor_name).device_name
    code = f"""
import torch
fn = torch.{device_name}
"""
    torch_device_object = get_codegen_result(code, "fn")
    return torch_device_object


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

    global vendor_module
    if vendor_module is None:
        vendor_module = get_module("_" + vendor_name)
    return vendor_module


def get_vendor_info(vendor_name=None, query=False):
    if query:
        return get_vendor_module(vendor_name, query).vendor_info
    global vendor_module  # noqa: F824
    get_vendor_module(vendor_name)
    return vendor_module.vendor_info


def get_vendor_infos():
    infos = []
    for vendor_name in vendors.get_all_vendors():
        vendor_name = "_" + vendor_name
        try:
            single_info = get_vendor_info(vendor_name, query=True)
            infos.append(single_info)
        except Exception:
            pass

    return infos


def get_current_device_extend_op(vendor_name=None):
    import_vendor_extra_lib(vendor_name)
    global customized_ops
    if customized_ops is not None:
        return customized_ops
    customized_ops = []
    if ops_module is not None:
        ops = inspect.getmembers(ops_module, inspect.isfunction)
        customized_ops += ops
    if fused_module is not None:
        fused_ops = inspect.getmembers(fused_module, inspect.isfunction)
        customized_ops += fused_ops
    return customized_ops


def get_curent_device_unused_op(vendor_name=None):
    global vendor_module  # noqa: F824
    get_vendor_module(vendor_name)
    return list(vendor_module.CUSTOMIZED_UNUSED_OPS)


def get_heuristic_config(vendor_name=None):
    global heuristic_config_module
    try:
        heuristic_config_module = importlib.import_module(
            f"_{vendor_name}.heuristics_config_utils"
        )
    except:  # noqa E722
        heuristic_config_module = importlib.import_module(
            "_nvidia.heuristics_config_utils"
        )
    if hasattr(heuristic_config_module, "HEURISTICS_CONFIGS"):
        return heuristic_config_module.HEURISTICS_CONFIGS
    return None


def get_tune_config(vendor_name=None):
    global vendor_module  # noqa: F824
    get_vendor_module(vendor_name)
    return backend_utils.get_tune_config(vendor_name)


__all__ = ["*"]
