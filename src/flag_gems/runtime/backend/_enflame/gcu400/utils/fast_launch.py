# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cached raw launcher for the host-bound Triton kernels on gcu400.

``kernel[grid](...)`` re-binds the Python argument list, recomputes the
specialisation key and walks the JIT launcher on every call.  On this backend
that costs ~40us for the 16-argument kernel used by ``cat`` while the kernel
itself finishes in a few microseconds, so the launch path - not the kernel -
decides the measured latency of every small copy operator.

``CompiledKernel.run`` takes an already-specialised artifact plus the raw
argument values (pointers as plain ints), so caching that artifact per argument
shape turns a repeat launch into one dict lookup plus the driver call (5.6us
measured for the same cat kernel).

The enflame launcher decl format keeps a slot for every signature entry, so the
positional payload must follow the *declaration order* of ``kernel.params`` with
constexpr values included at their own position.

The cache key carries every fact Triton specialises on: the kernel, the grid,
``num_warps``, the dtype and 16-byte alignment of each pointer, every scalar
value and every constexpr value.  A cache hit is therefore compiled for exactly
these arguments and never reuses a handle built for a different alignment or a
different scalar constant.

This mirrors the launch path ``ccol_indices_copy.py`` already uses; it is
extracted here so the copy / cat / lift kernels share one implementation.
"""

import torch

from flag_gems.runtime import torch_device_fn

__all__ = ["launch"]

_CACHE_MAX = 1024
# id(kernel) -> (is_pointer, is_constexpr, name) in kernel declaration order.
_LAYOUT_CACHE = {}
# (kernel, grid, num_warps, pointer signature, scalars, constexprs) -> entry.
_CACHE = {}

try:
    _get_current_stream = torch._C._accelerator_getStream
except AttributeError:  # pragma: no cover - older torch without the unified API

    def _get_current_stream(device_index):
        return torch.cuda.current_stream(device_index)


def _layout(kernel):
    """Return ``[(is_constexpr, name)]`` in declaration order."""
    layout = _LAYOUT_CACHE.get(id(kernel))
    if layout is None:
        layout = [(param.is_constexpr, param.name) for param in kernel.params]
        _LAYOUT_CACHE[id(kernel)] = layout
    return layout


def _positional_args(kernel, tensors, scalars):
    """The non-constexpr arguments in declaration order, for the JIT call."""
    return [
        tensors[name] if name in tensors else scalars[name]
        for is_constexpr, name in _layout(kernel)
        if not is_constexpr
    ]


def _build(kernel, grid, tensors, scalars, constexprs, num_warps, device):
    """Compile through the regular JIT path and keep the raw launch pieces."""
    ambient_device = torch.cuda.current_device()
    with torch_device_fn.device(device):
        compiled = kernel[grid](
            *_positional_args(kernel, tensors, scalars),
            num_warps=num_warps,
            **constexprs,
        )
    needs_device_guard = ambient_device != device.index
    # Raw payload template: pointer slots store their index into the per-call
    # pointer tuple, every other slot stores the constant for this cache key.
    template = []
    pointer_index = 0
    for is_constexpr, name in _layout(kernel):
        if is_constexpr:
            template.append((False, constexprs[name]))
        elif name in tensors:
            template.append((True, pointer_index))
            pointer_index += 1
        else:
            template.append((False, scalars[name]))
    return (
        compiled.run,
        compiled.function,
        compiled.packed_metadata,
        needs_device_guard,
        tuple(template),
    )


def launch(
    kernel,
    grid,
    tensors,
    scalars=None,
    constexprs=None,
    num_warps=4,
    device=None,
    cache=None,
):
    """Launch ``kernel`` on ``grid`` through the cached raw launcher.

    ``tensors``/``scalars``/``constexprs`` map parameter names to values and must
    cover exactly the kernel's parameters.  Falls back to the regular
    ``kernel[grid](...)`` path whenever the compiled artifact cannot be cached
    (for example a Triton build without ``CompiledKernel.run``).
    """
    if cache is None:
        cache = _CACHE
    scalars = scalars or {}
    constexprs = constexprs or {}
    grid = tuple(grid)
    if device is None:
        device = next(iter(tensors.values())).device

    # Pointer order follows the kernel declaration, not the caller's dict order.
    pointer_tensors = [
        tensors[name]
        for is_constexpr, name in _layout(kernel)
        if not is_constexpr and name in tensors
    ]
    pointer_ints = tuple(tensor.data_ptr() for tensor in pointer_tensors)
    key = (
        id(kernel),
        grid,
        num_warps,
        device.index,
        *((t.dtype, ptr & 15) for t, ptr in zip(pointer_tensors, pointer_ints)),
        *scalars.values(),
        *constexprs.values(),
    )

    entry = cache.get(key)
    if entry is None:
        try:
            entry = _build(
                kernel, grid, tensors, scalars, constexprs, num_warps, device
            )
        except Exception:  # any Triton API difference: keep the regular path
            entry = False
        if len(cache) >= _CACHE_MAX:
            cache.clear()
        cache[key] = entry
    if not entry:
        with torch_device_fn.device(device):
            kernel[grid](
                *_positional_args(kernel, tensors, scalars),
                num_warps=num_warps,
                **constexprs,
            )
        return

    run, function, packed_metadata, needs_device_guard, template = entry
    payload = [
        pointer_ints[value] if is_pointer else value for is_pointer, value in template
    ]
    stream = _get_current_stream(device.index).native_handle
    grid_x = grid[0]
    grid_y = grid[1] if len(grid) > 1 else 1
    grid_z = grid[2] if len(grid) > 2 else 1
    if needs_device_guard:
        with torch_device_fn.device(device):
            run(
                grid_x,
                grid_y,
                grid_z,
                stream,
                function,
                packed_metadata,
                None,
                None,
                None,
                *payload,
            )
    else:
        run(
            grid_x,
            grid_y,
            grid_z,
            stream,
            function,
            packed_metadata,
            None,
            None,
            None,
            *payload,
        )
