#!/usr/bin/env python3
"""Pre-build Triton kernels for C++ pointwise_dynamic dispatch.

This script bridges the Python codegen and the C++ runtime by:
  1. Importing FlagGems op modules (e.g. ``flag_gems.ops.add``)
  2. Discovering all ``@pointwise_dynamic`` decorated functions
  3. Invoking ``PointwiseDynamicFunction.get_kernel_info()`` for each tensor
     rank to trigger the *same* codegen used by the Python runtime
  4. Emitting two C++ headers consumed by the C++ build:
     - ``pointwise_manifest.h`` — kernel registry mapping
       (op_name, rank) -> file path, kernel name, schema metadata
     - ``pointwise_runtime.h`` — thin inline wrappers that call into
       ``dispatch_pointwise()`` defined in the hand-written
       ``pointwise_util.h``

Because the kernel ``.py`` files are produced by the shared codegen in
``flag_gems.utils.pointwise_dynamic``, the C++ path uses *exactly* the
same Triton kernels as the Python path — no duplicated codegen logic.

Usage
-----
Specify op modules explicitly::

    python prebuild_kernels.py \\
        --op-files flag_gems.ops.abs flag_gems.ops.add \\
        --max-rank 6 \\
        --output-dir build/pointwise_kernels

Or scan an entire directory::

    python prebuild_kernels.py \\
        --op-dir src/flag_gems/ops \\
        --max-rank 6 \\
        --output-dir build/pointwise_kernels

Typically invoked automatically by CMake (see ``CMakeLists.txt``).
"""

import argparse
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Path setup — ensure FlagGems source tree is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
FLAGGEMS_ROOT = SCRIPT_DIR.parent.parent
FLAGGEMS_SRC = FLAGGEMS_ROOT / "src"
sys.path.insert(0, str(FLAGGEMS_SRC))

try:
    from flag_gems.utils.code_cache import code_cache_dir
    from flag_gems.utils.pointwise_dynamic import PointwiseDynamicFunction
except ImportError as e:
    print(f"Error importing FlagGems: {e}")
    print(f"Ensure FlagGems source can be found at: {FLAGGEMS_SRC}")
    sys.exit(1)


# ===================================================================
# Data classes
# ===================================================================


@dataclass
class PromotionRule:
    """A single dtype-promotion rule extracted from ``FunctionSchema``.

    Attributes:
        arg_indices: Positional indices of the input arguments that
            participate in this promotion (e.g. ``[0, 1]`` for a binary
            op that promotes its two operands together).
        method: Name of the promotion strategy, mirroring
            ``ELEMENTWISE_TYPE_PROMOTION_KIND`` in PyTorch / FlagGems
            (e.g. ``"DEFAULT"``, ``"INT_TO_FLOAT"``, ``"ALWAYS_BOOL"``).
    """

    arg_indices: List[int]
    method: str


@dataclass
class KernelEntry:
    """Metadata for one generated Triton kernel (a specific op + rank).

    One ``KernelEntry`` is created per (op_name, rank) pair.  The C++
    manifest header is generated from a list of these entries.

    Attributes:
        op_name: The Python attribute name of the
            ``PointwiseDynamicFunction`` (e.g. ``"add_func"``).
        func_name: The ``__name__`` of the ``@triton.jit`` scalar
            function that the kernel calls (usually the same as
            *op_name*).
        rank: Tensor rank this kernel is specialised for (0 = scalar,
            1 = 1-D, …).
        file_path: Absolute path to the generated ``.py`` file in the
            code-cache directory.
        kernel_name: The Triton kernel symbol name inside that file
            (e.g. ``"add_func_kernel_rank_2_bptr"``).
        num_input_tensors: Number of tensor inputs in the op schema.
        num_non_tensor_inputs: Number of scalar (non-tensor) inputs.
        num_outputs: Number of output tensors.
        promotion_rules: Dtype-promotion rules for each output.
        is_1d_tile: Whether the kernel uses the 1D-tile codegen path
            (no stride_order args, single tile_size constexpr).
        is_block_pointer: Whether the kernel uses block pointers
            (stride_order args present in the kernel signature).
        max_tile_size: Maximum tile size from the codegen config (e.g.
            512 for NVIDIA).  Needed by the C++ dispatch to replicate
            the Python budget-based per-dimension tile-size heuristic.
    """

    op_name: str
    func_name: str
    rank: int
    file_path: str
    kernel_name: str
    num_input_tensors: int
    num_non_tensor_inputs: int
    num_outputs: int
    promotion_rules: List[PromotionRule]
    is_1d_tile: bool
    is_block_pointer: bool
    max_tile_size: int


# ===================================================================
# Module discovery helpers
# ===================================================================


def find_pointwise_functions(module) -> Dict[str, PointwiseDynamicFunction]:
    """Return all ``PointwiseDynamicFunction`` instances in *module*.

    The ``@pointwise_dynamic`` decorator turns a ``@triton.jit`` function
    into a ``PointwiseDynamicFunction``.  This helper inspects every
    public attribute of *module* and collects those instances.

    Args:
        module: An imported Python module object.

    Returns:
        A dict mapping attribute name -> ``PointwiseDynamicFunction``.
    """
    functions = {}
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, PointwiseDynamicFunction):
            functions[name] = obj
    return functions


def import_op_module(module_path: str):
    """Import a FlagGems op module by dotted path or filesystem path.

    Args:
        module_path: Either a Python dotted module path
            (``"flag_gems.ops.add"``) or an absolute/relative ``.py``
            file path (``"/path/to/add.py"``).

    Returns:
        The imported module object.
    """
    if module_path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("op_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_path)


def scan_op_directory(op_dir: Path) -> List[str]:
    """Discover all op modules under *op_dir*.

    Scans for ``.py`` files (skipping those starting with ``_``) and
    converts each path to a dotted module name relative to
    ``FLAGGEMS_SRC``.

    Args:
        op_dir: Directory to scan (e.g. ``src/flag_gems/ops``).

    Returns:
        A list of dotted module paths, e.g.
        ``["flag_gems.ops.abs", "flag_gems.ops.add", ...]``.
    """
    modules = []
    for py_file in op_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        try:
            rel_path = py_file.relative_to(FLAGGEMS_SRC)
            module_name = (
                str(rel_path.with_suffix("")).replace("/", ".").replace("\\", ".")
            )
            modules.append(module_name)
        except ValueError:
            modules.append(str(py_file))
    return modules


# ===================================================================
# Kernel generation
# ===================================================================


def prebuild_from_function(
    func_name: str,
    pw_func: PointwiseDynamicFunction,
    max_rank: int,
    verbose: bool = False,
) -> List[KernelEntry]:
    """Generate Triton kernels for every rank and collect metadata.

    For each rank in ``[0, max_rank]``, calls
    ``pw_func.get_kernel_info(rank)`` which triggers the shared Python
    codegen.  The resulting ``.py`` kernel file is written to the
    FlagGems code-cache directory, and its metadata is returned so
    that the C++ manifest can reference it.

    Args:
        func_name: The attribute name of the function in the op module.
        pw_func: The ``PointwiseDynamicFunction`` instance.
        max_rank: Maximum tensor rank to generate kernels for
            (inclusive).
        verbose: If ``True``, print per-rank details.

    Returns:
        A list of ``KernelEntry`` objects, one per successfully
        generated rank.
    """
    entries = []

    # Extract schema metadata (shared across all ranks)
    schema = pw_func.fx
    num_input_tensors = schema.num_input_tensors()
    num_non_tensor_inputs = schema.num_non_tensor_args()
    num_outputs = schema.num_output_tensors()

    promotion_rules = []
    for pm in schema._promotion_methods:
        *arg_indices, method_enum = pm
        promotion_rules.append(
            PromotionRule(
                arg_indices=list(arg_indices),
                method=method_enum.name,
            )
        )

    scalar_fn_name = pw_func._scalar_fn.__name__

    # Kernel variant flags (determined by codegen config at build time)
    is_1d_tile = pw_func.config.prefer_1d_tile
    is_block_pointer = pw_func.config.prefer_block_pointer and not is_1d_tile
    cfg_max_tile_size = pw_func.config.max_tile_size

    for rank in range(max_rank + 1):
        try:
            kernel_info = pw_func.get_kernel_info(rank)

            entry = KernelEntry(
                op_name=func_name,
                func_name=scalar_fn_name,
                rank=rank,
                file_path=kernel_info.file_path,
                kernel_name=kernel_info.kernel_name,
                num_input_tensors=num_input_tensors,
                num_non_tensor_inputs=num_non_tensor_inputs,
                num_outputs=num_outputs,
                promotion_rules=promotion_rules,
                is_1d_tile=is_1d_tile,
                is_block_pointer=is_block_pointer,
                max_tile_size=cfg_max_tile_size,
            )
            entries.append(entry)

            if verbose:
                print(f"      rank={rank}: kernel_name={kernel_info.kernel_name}")
                print(f"               file_path={kernel_info.file_path}")

        except Exception as e:
            print(f"    Warning: rank {rank} failed: {e}")

    return entries


# ===================================================================
# Report generation
# ===================================================================


def generate_kernel_report(
    entries: List[KernelEntry],
    ops_found: Dict[str, List[KernelEntry]],
) -> str:
    """Produce a human-readable report of all generated kernels.

    The report is written to ``kernel_report.txt`` for build-time
    verification — it is *not* consumed by C++ code.

    Args:
        entries: Flat list of all kernel entries.
        ops_found: Dict mapping op name -> list of entries for that op.

    Returns:
        The report as a multi-line string.
    """
    lines = [
        "=" * 80,
        "KERNEL GENERATION REPORT",
        "=" * 80,
        "",
        f"Total operations: {len(ops_found)}",
        f"Total kernels: {len(entries)}",
        "",
    ]

    for op_name in sorted(ops_found.keys()):
        op_entries = ops_found[op_name]
        if not op_entries:
            continue

        sample = op_entries[0]
        lines.append("-" * 80)
        lines.append(f"Operation: {op_name}")
        lines.append(f"  Scalar function: {sample.func_name}")
        lines.append(f"  Num input tensors: {sample.num_input_tensors}")
        lines.append(f"  Num non-tensor inputs: {sample.num_non_tensor_inputs}")
        lines.append(f"  Num outputs: {sample.num_outputs}")
        lines.append("  Promotion rules:")
        for i, rule in enumerate(sample.promotion_rules):
            lines.append(f"    Output {i}: args{rule.arg_indices} -> {rule.method}")
        lines.append("")
        lines.append("  Kernels:")

        for entry in sorted(op_entries, key=lambda e: e.rank):
            lines.append(f"    Rank {entry.rank}:")
            lines.append(f"      kernel_name: {entry.kernel_name}")
            lines.append(f"      file_path:   {entry.file_path}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


# ===================================================================
# C++ header generation
# ===================================================================


def _format_promotion_rules_cpp(rules: List[PromotionRule]) -> str:
    """Serialise promotion rules as a C++ initializer-list literal.

    Example output::

        {{0, 1}, TypePromotionKind::DEFAULT}

    Args:
        rules: List of ``PromotionRule`` to format.

    Returns:
        A brace-enclosed C++ initializer string, or ``"{}"`` if empty.
    """
    if not rules:
        return "{}"
    parts = []
    for rule in rules:
        indices_str = ", ".join(str(i) for i in rule.arg_indices)
        parts.append(f"{{{{{indices_str}}}, TypePromotionKind::{rule.method}}}")
    return "{" + ", ".join(parts) + "}"


def generate_manifest_header(entries: List[KernelEntry], max_rank: int) -> str:
    """Generate ``pointwise_manifest.h`` — the C++ kernel registry.

    The header defines:
    - ``TypePromotionKind`` enum (mirrors Python's
      ``ELEMENTWISE_TYPE_PROMOTION_KIND``)
    - ``PromotionRule`` / ``KernelInfo`` structs
    - ``KERNEL_REGISTRY`` — a nested map from
      ``(op_name, rank) -> KernelInfo``
    - ``get_kernel_info()`` — lookup helper

    Args:
        entries: All kernel entries across all ops and ranks.
        max_rank: The maximum rank that was generated.

    Returns:
        The complete header file content as a string.
    """
    lines = [
        "#pragma once",
        "",
        "// ==========================================================================",
        "// Auto-generated by prebuild_kernels.py",
        "// Extracted from existing FlagGems op files",
        "// Uses the SAME codegen as Python pointwise_dynamic",
        "// ==========================================================================",
        "",
        "#include <string>",
        "#include <vector>",
        "#include <unordered_map>",
        "",
        "namespace pointwise_dynamic {",
        "",
        "// Mirrors Python's ELEMENTWISE_TYPE_PROMOTION_KIND",
        "enum class TypePromotionKind {",
        "    DEFAULT,",
        "    NO_OPMATH,",
        "    INT_TO_FLOAT,",
        "    ALWAYS_BOOL,",
        "    COMPLEX_TO_FLOAT,",
        "    BOOL_TO_LONG,",
        "};",
        "",
        "// Type promotion rule for one output: which input args to consider + method",
        "struct PromotionRule {",
        "    std::vector<int> arg_indices;",
        "    TypePromotionKind method;",
        "};",
        "",
        "struct KernelInfo {",
        "    std::string file_path;",
        "    std::string kernel_name;",
        "    int num_input_tensors;",
        "    int num_non_tensor_inputs;",
        "    int num_outputs;",
        "    std::vector<PromotionRule> promotion_rules;",
        "    bool is_1d_tile;       // 1D-tile kernel (no stride_order, single tile_size)",
        "    bool is_block_pointer;  // block-pointer kernel (stride_order present)",
        "    int max_tile_size;      // max tile size budget from codegen config",
        "};",
        "",
    ]

    # Group entries by op_name
    ops: Dict[str, List[KernelEntry]] = {}
    for entry in entries:
        ops.setdefault(entry.op_name, []).append(entry)

    lines.append("// Kernel registry: op_name -> rank -> KernelInfo")
    lines.append(
        "inline const std::unordered_map<std::string,"
        " std::unordered_map<int, KernelInfo>> KERNEL_REGISTRY = {"
    )

    for op_name in sorted(ops.keys()):
        op_entries = ops[op_name]
        lines.append(f'    {{"{op_name}", {{')
        for entry in sorted(op_entries, key=lambda e: e.rank):
            rules_str = _format_promotion_rules_cpp(entry.promotion_rules)
            is_1d = "true" if entry.is_1d_tile else "false"
            is_bptr = "true" if entry.is_block_pointer else "false"
            lines.append(
                f"        {{{entry.rank}, KernelInfo{{"
                f'"{entry.file_path}", "{entry.kernel_name}", '
                f"{entry.num_input_tensors}, {entry.num_non_tensor_inputs}, {entry.num_outputs}, "
                f"{rules_str}, "
                f"{is_1d}, {is_bptr}, {entry.max_tile_size}"
                f"}}}},"
            )
        lines.append("    }},")

    lines.append("};")
    lines.append("")
    lines.append(f"constexpr int MAX_RANK = {max_rank};")
    lines.append("")
    lines.append("// Lookup helper — returns nullptr when (op_name, rank) is not found")
    lines.append(
        "inline const KernelInfo* get_kernel_info(const std::string& op_name, int rank) {"
    )
    lines.append("    auto op_it = KERNEL_REGISTRY.find(op_name);")
    lines.append("    if (op_it == KERNEL_REGISTRY.end()) return nullptr;")
    lines.append("    auto rank_it = op_it->second.find(rank);")
    lines.append("    if (rank_it == op_it->second.end()) return nullptr;")
    lines.append("    return &rank_it->second;")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace pointwise_dynamic")

    return "\n".join(lines)


def _build_is_tensor_mask(num_tensors: int, num_scalars: int) -> str:
    """Build the ``is_tensor_mask`` C++ initializer from counts.

    The mask reflects the *original* argument order in the Python
    ``@pointwise_dynamic`` schema.  For simplicity, we assume tensors
    come first, followed by scalars — which matches how
    ``FunctionSchema`` orders them for all standard FlagGems ops.

    Args:
        num_tensors: Number of tensor inputs.
        num_scalars: Number of scalar (non-tensor) inputs.

    Returns:
        A brace-enclosed C++ bool initializer, e.g. ``"{true, true, false}"``.
    """
    parts = ["true"] * num_tensors + ["false"] * num_scalars
    return "{" + ", ".join(parts) + "}"


def _generate_wrapper(op_name: str, num_tensors: int, num_scalars: int) -> str:
    """Generate a pair of inline C++ wrappers for a single op.

    Produces both a normal version (allocates output) and an ``_out``
    version (caller provides pre-allocated output), matching the
    ``dispatch_pointwise`` / ``dispatch_pointwise_out`` signatures in
    ``pointwise_util.h``.

    Args:
        op_name: The op name used as the registry key.
        num_tensors: Number of tensor inputs.
        num_scalars: Number of scalar (non-tensor) inputs.

    Returns:
        A C++ code string containing both wrapper functions, or an
        empty string with a comment if the combination is unsupported.

    Supported combinations (nt=num_tensors, ns=num_scalars):
        - (1,0) Unary
        - (2,0) Binary
        - (1,1) Unary + 1 scalar
        - (2,1) Binary + 1 scalar
        - (1,2) Unary + 2 scalars
        - (3,0) Ternary
        - (3,1) Ternary + 1 scalar
    """
    nt, ns = num_tensors, num_scalars
    mask = _build_is_tensor_mask(nt, ns)

    # --- tensor parameter lists ---
    tensor_params = {
        1: "const at::Tensor& x",
        2: "const at::Tensor& a, const at::Tensor& b",
        3: "const at::Tensor& a, const at::Tensor& b, const at::Tensor& c",
    }
    tensor_args = {
        1: "{x}",
        2: "{a, b}",
        3: "{a, b, c}",
    }

    if nt not in tensor_params:
        return f"// TODO: unsupported combination nt={nt}, ns={ns} for {op_name}\n\n"

    t_params = tensor_params[nt]
    t_args = tensor_args[nt]

    # --- scalar parameter / argument lists ---
    if ns == 0:
        s_params_with_comma = ""
        s_args = "{}"
    elif ns == 1:
        s_params_with_comma = ", double scalar = 1.0"
        s_args = "{scalar}"
    elif ns == 2:
        s_params_with_comma = ", double scalar0 = 0.0, double scalar1 = 0.0"
        s_args = "{scalar0, scalar1}"
    else:
        return f"// TODO: unsupported {ns} scalars for {op_name}\n\n"

    code = ""

    # Helper: resolve the op registry once via static local, then delegate
    # to dispatch_pointwise_impl which takes the pre-resolved registry.
    # This eliminates the string hash map lookup on every call.
    registry_lookup = (
        f'    static const auto& registry = KERNEL_REGISTRY.at("{op_name}");\n'
    )

    # Normal wrapper
    all_params = t_params + (s_params_with_comma if ns else "")
    code += f"inline at::Tensor {op_name}({all_params}) {{\n"
    code += registry_lookup
    code += f"    return dispatch_pointwise_impl(registry, {t_args}, {s_args}, {mask}, {{}});\n"
    code += "}\n\n"

    # _out wrapper (out tensor inserted after input tensors)
    out_params = t_params + ", at::Tensor& out" + (s_params_with_comma if ns else "")
    code += f"inline at::Tensor {op_name}_out({out_params}) {{\n"
    code += registry_lookup
    code += "    std::vector<c10::optional<at::Tensor>> pre_outputs = {out};\n"
    code += f"    return dispatch_pointwise_impl(registry, {t_args}, {s_args}, {mask}, pre_outputs);\n"
    code += "}\n\n"

    return code


def generate_runtime_header(ops: Dict[str, List[KernelEntry]]) -> str:
    """Generate ``pointwise_runtime.h`` — per-op inline C++ wrappers.

    Each discovered op gets two thin wrappers:

    - ``<op_name>(tensors..., scalars...)`` — allocates a new output
      tensor via ``dispatch_pointwise``.
    - ``<op_name>_out(tensors..., out, scalars...)`` — writes into a
      caller-provided output via ``dispatch_pointwise_out``.

    All heavy-lifting (broadcasting, dtype promotion, StridedBuffer
    wrapping, kernel lookup/launch) is delegated to the hand-written
    ``pointwise_util.h``.

    Args:
        ops: Dict mapping op name -> list of ``KernelEntry`` (used only
            to read ``num_input_tensors`` / ``num_non_tensor_inputs``).

    Returns:
        The complete header file content as a string.
    """
    header = """\
#pragma once

// ==========================================================================
// Auto-generated operator wrappers for pointwise_dynamic operations.
// Generated by prebuild_kernels.py -- DO NOT EDIT.
//
// Stable utilities (broadcast, type promotion, dispatch) are in
// pointwise_util.h which is included transitively.
// ==========================================================================

#include "pointwise_prepare_args.h"

namespace pointwise_dynamic {

"""

    for op_name, op_entries in sorted(ops.items()):
        if not op_entries:
            continue
        sample = op_entries[0]
        header += _generate_wrapper(
            op_name, sample.num_input_tensors, sample.num_non_tensor_inputs
        )

    header += "}  // namespace pointwise_dynamic\n"
    return header


# ===================================================================
# Main entry point
# ===================================================================


def main():
    """Parse arguments, generate kernels, and emit C++ headers."""
    parser = argparse.ArgumentParser(
        description="Pre-build pointwise kernels from FlagGems op files"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--op-files",
        nargs="+",
        type=str,
        help="Module paths (e.g. flag_gems.ops.abs flag_gems.ops.add)",
    )
    input_group.add_argument(
        "--op-dir",
        type=str,
        help="Directory containing op files to scan",
    )

    parser.add_argument("--max-rank", type=int, default=6, help="Maximum tensor rank")
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Pre-building pointwise kernels from FlagGems op files")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Cache directory: {code_cache_dir()}")
    print(f"Max rank: {args.max_rank}")
    print()

    all_entries: List[KernelEntry] = []
    ops_found: Dict[str, List[KernelEntry]] = {}

    # Determine which modules to process
    if args.op_files:
        modules_to_process = [m.strip() for m in args.op_files]
        print(f"Processing specified modules: {modules_to_process}")
    elif args.op_dir:
        op_dir = Path(args.op_dir)
        if not op_dir.exists():
            print(f"Error: Directory not found: {op_dir}")
            sys.exit(1)
        modules_to_process = scan_op_directory(op_dir)
        print(f"Scanned {len(modules_to_process)} modules from {op_dir}")
    else:
        modules_to_process = []

    print()

    # Process each module
    for module_path in modules_to_process:
        print(f"Processing: {module_path}")

        try:
            module = import_op_module(module_path)
        except Exception as e:
            print(f"  Error importing: {e}")
            continue

        pw_functions = find_pointwise_functions(module)

        if not pw_functions:
            print("  No @pointwise_dynamic functions found")
            continue

        print(
            f"  Found {len(pw_functions)} pointwise functions: {list(pw_functions.keys())}"
        )

        for func_name, pw_func in pw_functions.items():
            print(f"  Building {func_name}...")

            entries = prebuild_from_function(
                func_name, pw_func, args.max_rank, verbose=args.verbose
            )

            if entries:
                all_entries.extend(entries)
                ops_found[func_name] = entries
                print(f"    Generated {len(entries)} kernels (ranks 0-{args.max_rank})")
            else:
                print("    No kernels generated")

    # -----------------------------------------------------------------
    # Emit C++ headers and report
    # -----------------------------------------------------------------
    print()
    print("=" * 60)
    print("Generating C++ headers")
    print("=" * 60)

    manifest_path = output_dir / "pointwise_manifest.h"
    manifest_path.write_text(generate_manifest_header(all_entries, args.max_rank))
    print(f"Generated: {manifest_path}")

    runtime_path = output_dir / "pointwise_runtime.h"
    runtime_path.write_text(generate_runtime_header(ops_found))
    print(f"Generated: {runtime_path}")

    report_path = output_dir / "kernel_report.txt"
    report_path.write_text(generate_kernel_report(all_entries, ops_found))
    print(f"Generated: {report_path}")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total operations: {len(ops_found)}")
    print(f"Total kernels: {len(all_entries)}")
    print(f"Operations: {', '.join(sorted(ops_found.keys()))}")
    print()
    print("Kernel files are in:", code_cache_dir())
    print()
    print("Generated files:")
    print(f"  {manifest_path.name}  - C++ kernel registry header")
    print(f"  {runtime_path.name}   - C++ runtime wrappers")
    print(f"  {report_path.name}    - Detailed kernel report (for verification)")
    print()
    print("Usage in C++:")
    print(f'  #include "{manifest_path.name}"')
    print(f'  #include "{runtime_path.name}"')
    print("  auto result = pointwise_dynamic::add_func(a, b, alpha);")


if __name__ == "__main__":
    main()
