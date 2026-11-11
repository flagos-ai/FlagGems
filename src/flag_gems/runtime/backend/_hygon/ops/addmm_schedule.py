# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep a gfx936 addmm prefetch asynchronous across its LDS-only barrier."""

import functools
import hashlib
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

import triton
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend
from triton.runtime.cache import get_cache_manager


def _registers(text):
    result = set()
    for start, end in re.findall(r"v\[(\d+):(\d+)\]", text):
        result.update(range(int(start), int(end) + 1))
    result.update(map(int, re.findall(r"\bv(\d+)\b", text)))
    return result


def _defer_prefetch_wait(assembly):
    """Return unchanged code unless every dependency of this loop is verified."""
    lines = assembly.splitlines(keepends=True)
    headers = [i for i, line in enumerate(lines) if "Inner Loop Header" in line]
    if len(headers) != 1:
        return assembly
    header = headers[0]
    label = lines[header].split(":")[0]
    branches = [
        i
        for i in range(header + 1, len(lines))
        if re.match(r"\s*s_cbranch_scc1 " + re.escape(label) + r"\s*$", lines[i])
    ]
    if len(branches) != 1:
        return assembly
    body = range(header + 1, branches[0])
    waits = [i for i in body if lines[i].strip() == "s_waitcnt vmcnt(0) lgkmcnt(0)"]
    loads = [i for i in body if re.match(r"\s*buffer_load_dwordx4 ", lines[i])]
    stores = [i for i in body if re.match(r"\s*ds_write_b128 ", lines[i])]
    barriers = [i for i in body if lines[i].strip() == "s_barrier"]
    global_memory = [
        i for i in body if re.match(r"\s*(buffer|flat|global|image)_", lines[i])
    ]
    if (
        len(waits) != 2
        or len(loads) != 4
        or len(stores) != 4
        or len(barriers) != 2
        or global_memory != loads
        or not loads[0] < waits[0] < barriers[0] < loads[1]
        or not loads[-1] < waits[1] < barriers[1] < stores[0]
    ):
        return assembly
    destination = _registers(lines[loads[0]].split(",")[0])
    if len(destination) != 4:
        return assembly
    if any(
        destination.intersection(_registers(lines[i]))
        for i in range(loads[0] + 1, stores[0])
    ):
        return assembly
    if not destination <= _registers(lines[stores[0]].split(",", 1)[1]):
        return assembly
    # The existing final VM wait still precedes the only consumer. Preserve
    # both LDS barriers and every arithmetic instruction and accumulation order.
    lines[waits[0]] = lines[waits[0]].replace("vmcnt(0) ", "")
    return "".join(lines)


@functools.lru_cache(maxsize=128)
def _compile(kernel, m, n, k, target, device, attributes):
    attrs = {}
    for index, (aligned, small_storage) in enumerate(attributes):
        values = []
        if aligned:
            values.append(["tt.divisibility", 16])
        if small_storage:
            values.append(["tt.pointer_range", 32])
        if values:
            attrs[(index,)] = values
    options = dict(
        num_warps=16,
        num_stages=2,
        matrix_instr_nonkdim=16,
        kpack=2,
        optimize_epilogue=False,
    )
    compiled = triton.compile(
        ASTSource(
            kernel,
            signature={name: "*bf16" for name in ("A", "B", "I", "O")},
            constexprs=dict(M=m, N=n, K=k, BM=256, BN=256, BK=32),
            attrs=attrs,
        ),
        target=target,
        options=options,
    )
    assembly = compiled.asm["amdgcn"]
    scheduled = _defer_prefetch_wait(assembly)
    if scheduled == assembly:
        return compiled
    digest = hashlib.sha256((compiled.hash + scheduled).encode()).hexdigest()
    cache = get_cache_manager(digest)
    filename = "addmm.hsaco"
    binary_path = cache.get_file(filename)
    try:
        if binary_path is None:
            backend = make_backend(target)
            flags = [
                flag
                for flag in backend._get_clang_args(
                    compiled.metadata._asdict(), backend.parse_options(options)
                )
                if flag
            ]
            target_name = re.search(
                r'\.amdgcn_target "[^"]*--([^"]+)"', scheduled
            ).group(1)
            flags = [
                "-mcpu=" + target_name if flag.startswith("-mcpu=") else flag
                for flag in flags
            ]
            with tempfile.TemporaryDirectory(prefix="flaggems-addmm-") as directory:
                source = Path(directory) / "kernel.s"
                output = Path(directory) / filename
                source.write_text(scheduled)
                subprocess.run(
                    [
                        backend.path_to_rocm_clang(),
                        *flags,
                        "-x",
                        "assembler",
                        str(source),
                        "-o",
                        str(output),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                binary_path = cache.put(output.read_bytes(), filename, binary=True)
        binary = Path(binary_path).read_bytes()
    except (OSError, subprocess.CalledProcessError) as error:
        warnings.warn(
            f"Hygon addmm scheduling unavailable; using original kernel: {error}"
        )
        return compiled
    # This is a private CompiledKernel instance; no backend, JIT hook, shared
    # compiler installation, or original compilation-cache entry is modified.
    compiled.kernel = binary
    compiled.asm["hsaco"] = binary
    compiled.asm["amdgcn"] = scheduled
    compiled.hash = digest
    compiled.metadata = compiled.metadata._replace(hash=digest)
    return compiled


def launch(kernel, a, b, bias, out):
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        return False
    tensors = (a, b, bias, out)
    attributes = tuple(
        (tensor.data_ptr() % 16 == 0, tensor.untyped_storage().nbytes() <= 2**31 - 1)
        for tensor in tensors
    )
    m, k = a.shape
    n = b.shape[1]
    compiled = _compile(
        kernel,
        m,
        n,
        k,
        target,
        triton.runtime.driver.active.get_current_device(),
        attributes,
    )
    compiled[(triton.cdiv(m, 256) * triton.cdiv(n, 256), 1, 1)](a, b, bias, out)
    return True
