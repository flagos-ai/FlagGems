from .. import arch_version
from .hadamard_transform import hadamard_transform  # noqa: F401
from .hadamard_transform import hadamard_transform_12N  # noqa: F401
from .hadamard_transform import hadamard_transform_20N  # noqa: F401
from .hadamard_transform import hadamard_transform_28N  # noqa: F401
from .hadamard_transform import hadamard_transform_40N  # noqa: F401

_hadamard_exports = [
    "hadamard_transform",
    "hadamard_transform_12N",
    "hadamard_transform_20N",
    "hadamard_transform_28N",
    "hadamard_transform_40N",
]

__all__ = _hadamard_exports[:]

if arch_version == 300:
    from .gcu300.concat_and_cache_mla import concat_and_cache_mla  # noqa: F401
    from .gcu300.cross_entropy_loss import cross_entropy_loss
    from .gcu300.flash_mla import flash_mla
    from .gcu300.fused_add_rms_norm import fused_add_rms_norm
    from .gcu300.gelu_and_mul import gelu_and_mul
    from .gcu300.rotary_embedding import apply_rotary_pos_emb  # noqa: F401
    from .gcu300.silu_and_mul import silu_and_mul
    from .gcu300.skip_layernorm import skip_layer_norm

    __all__ += [
        "apply_rotary_pos_emb",
        "concat_and_cache_mla",
        "cross_entropy_loss",
        "flash_mla",
        "fused_add_rms_norm",
        "gelu_and_mul",
        "silu_and_mul",
        "skip_layer_norm",
    ]
elif arch_version == 400 or arch_version == 410:
    from .gcu400.cross_entropy_loss import cross_entropy_loss
    from .gcu400.flash_mla import flash_mla
    from .gcu400.fused_add_rms_norm import fused_add_rms_norm
    from .gcu400.gelu_and_mul import gelu_and_mul
    from .gcu400.outer import outer
    from .gcu400.silu_and_mul import silu_and_mul
    from .gcu400.skip_layernorm import skip_layer_norm

    __all__ = [
        "cross_entropy_loss",
        "flash_mla",
        "fused_add_rms_norm",
        "gelu_and_mul",
        "outer",
        "silu_and_mul",
        "skip_layer_norm",
    ]
