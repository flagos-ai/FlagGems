"""Public compatibility surface for the active Hopper FA3 TLE kernels."""

from .utils import (
    TLE_FA3_AVAILABLE,
    fa3_tle_paged_gather_mode,
    fa3_tle_paged_gather_name,
)
from .planning import FA3MetadataDispatch, fa3_tle_metadata_dispatch
from .fa_hopper_persistent_pingpong import flash_varlen_fwd_v3_tle_kernel
from .fa_hopper_short import flash_varlen_fwd_v3_tle_short_kernel
from .fa_hopper_direct import flash_varlen_fwd_v3_tle_direct_kernel
from .fa_hopper_decode_flashdecoding import (
    flash_varlen_fwd_v3_tle_decode_flashdecoding_combine_kernel,
    flash_varlen_fwd_v3_tle_decode_flashdecoding_kernel,
)

__all__ = [
    "TLE_FA3_AVAILABLE",
    "FA3MetadataDispatch",
    "fa3_tle_metadata_dispatch",
    "fa3_tle_paged_gather_mode",
    "fa3_tle_paged_gather_name",
    "flash_varlen_fwd_v3_tle_kernel",
    "flash_varlen_fwd_v3_tle_short_kernel",
    "flash_varlen_fwd_v3_tle_direct_kernel",
    "flash_varlen_fwd_v3_tle_decode_flashdecoding_kernel",
    "flash_varlen_fwd_v3_tle_decode_flashdecoding_combine_kernel",
]
