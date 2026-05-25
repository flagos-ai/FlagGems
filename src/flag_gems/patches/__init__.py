from flag_gems.patches.flag_ops import enable_flag_ops
from flag_gems.patches.patch_vllm_all import apply_gems_patches_to_vllm

__all__ = [
    "apply_gems_patches_to_vllm",
    "enable_flag_ops",
]

assert __all__ == sorted(__all__)
