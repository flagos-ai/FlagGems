import torch

vllm_C_lib = torch.library.Library("_C", "IMPL")
vllm_moe_C_lib = torch.library.Library("_moe_C", "IMPL")
vllm_fa3_C_lib = torch.library.Library("_vllm_fa3_C", "IMPL")

libs = {
    "_C": vllm_C_lib,
    "_moe_C": vllm_moe_C_lib,
    "_vllm_fa3_C": vllm_fa3_C_lib,
}


def patch_module_method(cls, method_name: str, new_method: callable, verbose=True):
    old_method = getattr(cls, method_name, None)
    setattr(cls, method_name, new_method)
    if verbose:
        print(
            f"Patched {cls.__name__}.{method_name} with FLAGGEMS {new_method.__name__}"
        )
    return old_method  # incase we need to revert the patch later


def patch_vllm_lib(lib_name, fn_name, fn, key, verbose=True):
    if lib_name not in libs:
        raise ValueError(f"Library {lib_name} is not recognized.")

    lib = libs[lib_name]
    lib.impl(fn_name, fn, key)

    if verbose:
        print(f"Patched torch.ops.{lib_name}.{fn_name} with FLAGGEMS {fn.__name__}")
