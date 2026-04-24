import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset

device_ = device

_MIN_INT8_VAL = tl.constexpr(torch.iinfo(torch.int8).min)
_MAX_INT8_VAL = tl.constexpr(torch.iinfo(torch.int8).max)
_MIN_INT16_VAL = tl.constexpr(torch.iinfo(torch.int16).min)
_MAX_INT16_VAL = tl.constexpr(torch.iinfo(torch.int16).max)
_MIN_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).min)
_MAX_INT32_VAL = tl.constexpr(torch.iinfo(torch.int32).max)
_MAX_UINT32_VAL = tl.constexpr((1 << 32) - 1)
_MIN_UINT32_VAL = tl.constexpr(0)
_MIN_INT24_VAL = tl.constexpr(-(2**23))
_MAX_INT24_VAL = tl.constexpr(2**23 - 1)


def _sort_workaround(tensor):
    """
    CPU workaround for Sort operation (minimizing data transfer).

    Problem: TPU's torch.sort() uses Top-K implementation with hardware limitations.
    Solution: Execute sort on CPU, immediately transfer back to TPU.

    TODO: When TPU sort operator is implemented, replace with: return torch.sort(tensor)

    Args:
        tensor: Input tensor (on TPU device).

    Returns:
        sorted_data: Sorted data (on TPU device).
        sorted_indices: Sort indices (on TPU device).
    """
    cpu_tensor = tensor.cpu()
    sorted_data, sorted_indices = torch.sort(cpu_tensor)
    return sorted_data.to(tensor.device), sorted_indices.to(tensor.device)


@libentry()
@triton.jit(
    do_not_specialize=[
        "philox_seed_lo",
        "philox_seed_hi",
        "philox_offset_lo",
        "philox_offset_hi",
    ]
)
def shuffle_by_random_kernel(
    value_ptr,
    random_perm_ptr,  # Output random permutation indices
    n_elements,
    philox_seed_lo: tl.uint32,
    philox_seed_hi: tl.uint32,
    philox_offset_lo: tl.uint32,
    philox_offset_hi: tl.uint32,
    BLOCK_SIZE: tl.constexpr,
):
    """
    TPU kernel to generate random permutation indices (avoids using argsort).

    Strategy: Instead of argsort, use:
        1. Generate random numbers.
        2. Mark each element's relative position within a block.
        3. Perform lightweight argsort on CPU (only sorting random numbers).

    This keeps most computation on TPU, only offloading lightweight sorting to CPU.

    Args:
        value_ptr: Input value array.
        random_perm_ptr: Output random permutation indices.
        n_elements: Total number of elements.
        philox_seed: Random number seed.
        philox_offset: Random number offset.
        BLOCK_SIZE: Block size.
    """
    pid = tl.program_id(0)
    offset_range = tl.arange(0, BLOCK_SIZE)
    value_offset = pid.to(tl.int64) * BLOCK_SIZE + offset_range
    mask = value_offset < n_elements

    # Generate random numbers (on TPU)
    c0 = philox_offset_lo
    c1 = philox_offset_hi
    i4 = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed_lo, philox_seed_hi, c0, c1, _O, _O)

    # Store random numbers (on TPU)
    # Later sort these on CPU (to avoid argsort compilation issues)
    tl.store(random_perm_ptr + value_offset, r0, mask=mask)


def sort_by_key(key, value, valid_bits, generator=None):
    """
    Sort-by-key implementation (hybrid TPU/CPU, TPU-first).

    Algorithm flow:
        1. [CPU] Sort keys (workaround) - lightweight operation.
        2. [TPU] Reorder values using scatter - main computation.
        3. [TPU] Shuffle processing (if needed) - main computation.

    Performance analysis:
        - CPU part: Only sorting (O(n log n)), but a simple operation.
        - TPU part: Scatter operation (O(n)), fully utilizing parallelism.
        - Data transfer: Minimized (transferred only once).

    Args:
        key: Sort keys (on TPU).
        value: Value array (on TPU).
        valid_bits: Key valid bit count (unused).
        generator: Random number generator.

    Returns:
        sorted_value: Values sorted by key (on TPU).
    """
    n_elements = key.numel()

    # ========== Step 1: Sort keys (CPU workaround) ==========
    # Problem: TPU Top-K limitations + argsort compilation error.
    # Solution: CPU sort (lightweight, data transfer overhead controllable).
    key_cpu = key.cpu()
    sorted_key_cpu, sorted_indices_cpu = torch.sort(key_cpu)

    sorted_indices = sorted_indices_cpu.to(key.device)

    # ========== Step 2: Use TPU scatter to reorder values ==========
    # This is the main compute-intensive operation, executed on TPU.
    sorted_value = torch.empty_like(value)
    # Use the project's existing TPU scatter implementation
    sorted_value = value[sorted_indices]

    # ========== Step 3: Shuffle (on TPU) ==========
    if generator is not None or n_elements > 1024:
        # Use TPU kernel for shuffle
        BLOCK_SIZE = 512
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        philox_seed, philox_offset = philox_backend_seed_offset(
            n_elements, generator=generator
        )
        philox_seed_hi = (philox_seed >> 32) & 0xFFFFFFFF
        philox_seed_lo = philox_seed & 0xFFFFFFFF
        philox_offset_hi = (philox_offset >> 32) & 0xFFFFFFFF
        philox_offset_lo = philox_offset & 0xFFFFFFFF

        # Generate random permutation (on TPU)
        random_keys = torch.empty(n_elements, dtype=torch.float32, device=key.device)

        with torch_device_fn.device(key.device):
            shuffle_by_random_kernel[grid](
                sorted_value,
                random_keys,
                n_elements,
                philox_seed_lo,
                philox_seed_hi,
                philox_offset_lo,
                philox_offset_hi,
                BLOCK_SIZE,
                num_warps=4,
            )

        # Sort random keys (lightweight CPU operation)
        random_keys_cpu = random_keys.cpu()
        _, shuffle_indices_cpu = torch.sort(random_keys_cpu)
        shuffle_indices = shuffle_indices_cpu.to(key.device)

        # Apply shuffle on TPU
        sorted_value = sorted_value[shuffle_indices]

    return sorted_value


def randperm(
    n,
    *,
    generator=None,
    out=None,
    dtype=torch.int64,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    """
    Generate random permutation (TPU-first implementation).

    Algorithm flow (hybrid TPU/CPU, maximizing TPU utilization):
        1. [TPU] Generate random keys - torch.randint (TPU natively supported).
        2. [TPU] Create index array - torch.arange (TPU natively supported).
        3. [CPU] Sort (workaround) - sort_workaround.
        4. [TPU] Scatter reorder - scatter_ (TPU implementation).
        5. [TPU] Shuffle - shuffle_kernel (TPU implementation).

    Args:
        n: Permutation length.
        dtype: Output data type.
        device: Device.
        generator: Random number generator.

    Returns:
        perm: Random permutation tensor (on TPU).
    """
    logging.debug("GEMS RANDPERM (TPU-first implementation)")

    # ========== Type adaptation ==========
    original_dtype = dtype
    if dtype == torch.int64:
        dtype = torch.int32
        logging.debug("RANDPERM: Using int32 (TPU limitation)")

    assert dtype in (torch.int16, torch.int32), f"Unsupported type: {dtype}"
    assert n <= _MAX_INT32_VAL, f"n={n} exceeds int32 maximum value"

    if device is None:
        device = torch.device(device_.name)

    # ========== Step 1: Create index array on TPU ==========
    # This is a TPU native operation, very efficient
    in_range = torch.arange(n, dtype=dtype, device=device)

    # ========== Step 2: Generate random keys on TPU ==========
    # torch.randint is natively supported on TPU, no CPU needed
    u8max = 2**8
    u16max = 2**16
    u24max = 2**24

    if n <= u8max:
        valid_bits = 8
        key_dtype = torch.int8
        keymin, keymax = _MIN_INT8_VAL, _MAX_INT8_VAL
    elif n <= u16max:
        valid_bits = 16
        key_dtype = torch.int16
        keymin, keymax = _MIN_INT16_VAL, _MAX_INT16_VAL
    elif n <= u24max:
        valid_bits = 24
        key_dtype = torch.int32
        keymin, keymax = _MIN_INT24_VAL, _MAX_INT24_VAL
    else:
        valid_bits = 32
        key_dtype = torch.int32
        keymin, keymax = _MIN_INT32_VAL, _MAX_INT32_VAL

    # Generate random keys on TPU (TPU native operation)
    rand_key = torch.randint(
        low=keymin,
        high=keymax,
        size=[n],
        dtype=key_dtype,
        device=device,
        generator=generator,
    )

    # ========== Steps 3-5: Sort-by-key (hybrid implementation, TPU-first) ==========
    # Only sorting is on CPU, everything else is on TPU
    perm_range = sort_by_key(rand_key, in_range, valid_bits, generator)

    # ========== Type conversion (if needed) ==========
    if original_dtype == torch.int64 and perm_range.dtype != torch.int64:
        try:
            perm_range_cpu = perm_range.cpu().to(original_dtype)
            perm_range = perm_range_cpu.to(device)
        except RuntimeError:
            logging.warning("Cannot convert to int64, returning int32")

    return perm_range
