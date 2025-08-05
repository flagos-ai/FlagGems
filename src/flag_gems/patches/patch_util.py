import torch

vllm_C_lib = torch.library.Library("_C", "IMPL")
vllm_moe_C_lib = torch.library.Library("_moe_C", "IMPL")


def patch_module_method(cls, method_name: str, new_method: callable, verbose=True):
    old_method = getattr(cls, method_name, None)
    setattr(cls, method_name, new_method)
    if verbose:
        print(
            f"Patched {cls.__name__}.{method_name} with FLAGGEMS {new_method.__name__}"
        )
    return old_method  # incase we need to revert the patch later


def patch_vllm_C_lib(name, fn, key, verbose=True):
    vllm_C_lib.impl(name, fn, key)
    if verbose:
        print(f"Patched torch.ops._C.{name} with FLAGGEMS {fn.__name__}")


def patch_vllm_moe_C_lib(name, fn, key, verbose=True):
    vllm_moe_C_lib.impl(name, fn, key)
    if verbose:
        print(f"Patched torch.ops._moe_C.{name} with FLAGGEMS {fn.__name__}")
