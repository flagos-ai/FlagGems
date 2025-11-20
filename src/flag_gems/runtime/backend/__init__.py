import ast
import functools
import importlib
import inspect
import os
import sys

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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def import_vendor_extra_lib(vendor_name=None, user_get=False):
    global vendor_extra_lib_imported
    if vendor_extra_lib_imported is True:
        return
    global ops_module, fused_module
    try:
        ops_module = importlib.import_module(f"_{vendor_name}.ops")
        if user_get:
            print(
                f"\033[92m[INFO]\033[0m : \033[92m operators of {vendor_name} vendor has been loaded\033[0m"
            )
    except Exception as err_msg:
        if user_get:
            print(
                f"\033[31m[Warning]\033[0m : \033[31mfailed to load operators of {vendor_name}"
                f"the reason is {err_msg}\033[0m"
            )
        else:
            print(
                f"\033[93m[Note]\033[0m :   No specialized common operators were found in"
                f" the {vendor_name} implementation, and general common operators are used by default."
            )
    except Exception as e:
        raise RuntimeError(f"Import vendor extra lib failed: {e}")

    try:
        fused_module = importlib.import_module(f"_{vendor_name}.fused")
    except ModuleNotFoundError:
        print(
            f"\033[93m[Note]\033[0m : No specialized fused operators were found in"
            f" the {vendor_name} implementation, and general fused operators are used by default."
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
        return get_module("_" + vendor_name)
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


def get_vendor_infos(return_type="list"):
    infos = {}
    import_failed_reasons = {}
    for vendor_name in vendors.get_all_vendors():
        try:
            single_info = get_vendor_info(vendor_name, query=True)
            infos.update({vendor_name: single_info})
        except Exception as err_msg:
            import_failed_reasons.update({vendor_name: err_msg})
    if return_type == "dict":
        return (infos, import_failed_reasons)
    else:
        return list(infos.values())


def get_current_device_extend_op(vendor_name=None, user_get=False):
    import_vendor_extra_lib(vendor_name, user_get)
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


def get_tune_config(vendor_name=None, query=False):
    global vendor_module  # noqa: F824
    if not query:
        get_vendor_module(vendor_name, query)
    return backend_utils.get_tune_config(vendor_name)


__all__ = ["*"]
