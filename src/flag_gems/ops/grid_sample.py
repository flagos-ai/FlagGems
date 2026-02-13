import logging
import warnings

import torch

from flag_gems import runtime

logger = logging.getLogger(__name__)

_ALIGN_CORNERS_WARNING = (
    "Default grid_sample and affine_grid behavior has changed "
    "to align_corners=False since 1.3.0. Please specify "
    "align_corners=True if the old behavior is desired. "
    "See the documentation of grid_sample for details."
)

try:
    _DISPATCH_KEY = torch._C._dispatch_key_parse(runtime.device.dispatch_key)
except Exception:
    _DISPATCH_KEY = None


class _DispatchKeyGuard:
    def __init__(self, key):
        self.key = key
        self.prev = None

    def __enter__(self):
        if self.key is None:
            return self
        self.prev = torch._C._dispatch_tls_is_dispatch_key_excluded(self.key)
        torch._C._dispatch_tls_set_dispatch_key_excluded(self.key, True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.key is None:
            return False
        torch._C._dispatch_tls_set_dispatch_key_excluded(self.key, self.prev)
        return False


def _normalize_mode(mode):
    if isinstance(mode, str):
        if mode == "bilinear":
            return 0
        if mode == "nearest":
            return 1
        if mode == "bicubic":
            return 2
        raise ValueError(
            "nn.functional.grid_sample(): expected mode to be "
            f"'bilinear', 'nearest' or 'bicubic', but got: '{mode}'"
        )
    if isinstance(mode, int):
        if mode not in (0, 1, 2):
            raise ValueError(
                "grid_sampler(): expected interpolation_mode to be 0, 1, or 2, "
                f"but got: {mode}"
            )
        return int(mode)
    raise TypeError(f"grid_sampler(): invalid interpolation_mode type: {type(mode)}")


def _normalize_padding_mode(padding_mode):
    if isinstance(padding_mode, str):
        if padding_mode == "zeros":
            return 0
        if padding_mode == "border":
            return 1
        if padding_mode == "reflection":
            return 2
        raise ValueError(
            "nn.functional.grid_sample(): expected padding_mode "
            "to be 'zeros', 'border', or 'reflection', "
            f"but got: '{padding_mode}'"
        )
    if isinstance(padding_mode, int):
        if padding_mode not in (0, 1, 2):
            raise ValueError(
                "grid_sampler(): expected padding_mode to be 0, 1, or 2, "
                f"but got: {padding_mode}"
            )
        return int(padding_mode)
    raise TypeError(f"grid_sampler(): invalid padding_mode type: {type(padding_mode)}")


def _normalize_align_corners(align_corners):
    if align_corners is None:
        warnings.warn(_ALIGN_CORNERS_WARNING)
        return False
    return bool(align_corners)


def _grid_sampler_impl(input, grid, interpolation_mode, padding_mode, align_corners):
    if interpolation_mode == 2 and input.dim() != 4:
        raise ValueError("grid_sampler(): bicubic mode is only supported for 4D input")
    with _DispatchKeyGuard(_DISPATCH_KEY):
        return torch.grid_sampler(
            input, grid, interpolation_mode, padding_mode, align_corners
        )


def grid_sampler(input, grid, interpolation_mode, padding_mode, align_corners):
    logger.debug("GEMS GRID_SAMPLER")
    mode_enum = _normalize_mode(interpolation_mode)
    padding_enum = _normalize_padding_mode(padding_mode)
    align_corners = _normalize_align_corners(align_corners)
    return _grid_sampler_impl(input, grid, mode_enum, padding_enum, align_corners)


def grid_sample(
    input,
    grid,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners=None,
):
    logger.debug("GEMS GRID_SAMPLE")
    mode_enum = _normalize_mode(mode)
    padding_enum = _normalize_padding_mode(padding_mode)
    align_corners = _normalize_align_corners(align_corners)
    return _grid_sampler_impl(input, grid, mode_enum, padding_enum, align_corners)
