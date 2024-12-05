import ast
import functools
import importlib
import os
import sys

from ..commom_utils import vendors_map
from . import backend_utils

vendor_module = None
device_name = None
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


def gen_torch_device_fn(api_name, vendor_name=None):
    global device_name
    device_name = device_name or get_vendor_info(vendor_name).device_name
    if api_name in device_fn_cache:
        return device_fn_cache[api_name]
    code = f"""
import torch
fn = torch.{device_name}.{api_name}
"""
    fn = get_codegen_result(code, "fn")
    device_fn_cache[api_name] = fn
    return fn


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


def get_vendor_infos() -> list:
    infos = []
    for vendor_name in vendors_map:
        vendor_name = "_" + vendor_name
        try:
            single_info = get_vendor_info(vendor_name, query=True)
            infos.append(single_info)
        except Exception:
            pass

    return infos


def get_curent_device_extend_op(vendor_name=None) -> dict:
    global vendor_module
    get_vendor_module(vendor_name)
    tuples = vendor_module.get_register_op_config()
    configs = {}
    for item in tuples:
        configs[item[0]] = item
    return configs


def get_curent_device_unused_op(vendor_name=None) -> list:
    global vendor_module
    get_vendor_module(vendor_name)
    return vendor_module.get_unused_op()


def get_tune_config(vendor_name=None) -> dict:
    global vendor_module
    get_vendor_module(vendor_name)
    return backend_utils.get_tune_config(vendor_name)


__all__ = ["*"]
