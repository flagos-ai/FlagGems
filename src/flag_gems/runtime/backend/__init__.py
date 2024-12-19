import ast
import functools
import importlib
import os
import sys

from ..commom_utils import vendors_map
from . import backend_utils

vendor_module = None
device_name = None
torch_device_object = None
torch_device_fn_device = None
tl_extra_backend_module = None
device_fn_cache = {}


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
    device_name = device_name or get_vendor_info(vendor_name).device_name
    module_str = f"triton.language.extra.{device_name}.libdevice"
    tl_extra_backend_module = importlib.import_module(module_str)


def get_tl_extra_backend_module():
    global tl_extra_backend_module
    return tl_extra_backend_module


def set_torch_backend_device_fn(vendor_name=None):
    global device_name, torch_device_fn_device
    device_name = device_name or get_vendor_info(vendor_name).device_name
    module_str = f"torch.backends.{device_name}"
    torch_device_fn_device = importlib.import_module(module_str)


def get_torch_backend_device_fn():
    global torch_device_fn_device
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
    global vendor_module
    get_vendor_module(vendor_name)
    return vendor_module.vendor_info


def get_vendor_infos():
    infos = []
    for vendor_name in vendors_map:
        vendor_name = "_" + vendor_name
        try:
            single_info = get_vendor_info(vendor_name, query=True)
            infos.append(single_info)
        except Exception:
            pass

    return infos


def get_curent_device_extend_op(vendor_name=None):
    global vendor_module
    get_vendor_module(vendor_name)
    return vendor_module.get_register_op_config()


def get_curent_device_unused_op(vendor_name=None):
    global vendor_module
    get_vendor_module(vendor_name)
    return list(vendor_module.get_unused_op())


def get_tune_config(vendor_name=None):
    global vendor_module
    get_vendor_module(vendor_name)
    return backend_utils.get_tune_config(vendor_name)


__all__ = ["*"]
