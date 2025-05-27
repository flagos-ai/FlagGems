from flag_gems.modules.normalization import gems_rms_forward
from flag_gems.patches.patch_util import patch_module_method


def custom_gems_rms_forward_cuda(self, x, residual=None):
    return gems_rms_forward(x, residual, self.weight, self.variance_epsilon)


def apply_gems_patches_to_vllm(verbose=True):
    from vllm.model_executor.layers.layernorm import RMSNorm

    patch_module_method(RMSNorm, "forward_cuda", custom_gems_rms_forward_cuda, verbose)
