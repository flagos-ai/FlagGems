#!/usr/bin/env python3
"""Extract operator metadata from FlagGems registrations, fused modules, and
predefined special sources.

The script parses ``src/flag_gems/__init__.py`` to recover the operator keys
used when constructing the ``Register`` instance, and combines them with the
symbols exported through ``src/flag_gems/fused/__init__.py``.  Each operator is
reported as a tuple ``(normalized_name, raw_list, source)`` where ``source`` is
``"aten"`` or ``"fused"``.  Normalization (enabled by default) applies the
following heuristics:

* drop any dotted suffix, e.g. ``add.Tensor`` -> ``add``
* drop a trailing ``_out`` suffix, treating ``mm_out`` as ``mm``
* drop a trailing ``_backward`` suffix, treating ``foo_backward`` as ``foo``
* strip leading underscores so private operators like ``_foo`` normalize to ``foo``
* remap selected canonical names via :data:`SPECIAL_NORMALIZE_MAP`

The script relies on the ``ast`` module for robustness against formatting
changes.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INIT_PATH = PROJECT_ROOT / "src" / "flag_gems" / "__init__.py"
FUSED_INIT_PATH = PROJECT_ROOT / "src" / "flag_gems" / "fused" / "__init__.py"


def _collect_registration_pairs(init_module: Path) -> Sequence[Tuple[str, str]]:
    """Parse ``flag_gems/__init__.py`` and extract operator registration tuples.

    Returns a sequence of ``(op_key, target_repr)`` pairs.  ``target_repr`` is the
    textual representation of the registered object (useful for debugging /
    cross-checking) but may be ``"<unknown>"`` when the AST node is not a simple
    ``Name``.
    """

    module_ast = ast.parse(init_module.read_text())

    class _EnableVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.registration_args: ast.AST | None = None

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            if node.name != "enable":
                return
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                if not stmt.targets:
                    continue
                target = stmt.targets[0]
                if (
                    not isinstance(target, ast.Name)
                    or target.id != "current_work_registrar"
                ):
                    continue
                if isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    if call.args:
                        self.registration_args = call.args[0]
            # No need to traverse deeper in ``enable`` once found.

    visitor = _EnableVisitor()
    visitor.visit(module_ast)

    if visitor.registration_args is None:
        raise RuntimeError(
            "Unable to locate the registration tuple in flag_gems.enable; "
            "has the initialization code changed?"
        )

    tuple_node = visitor.registration_args
    if not isinstance(tuple_node, (ast.List, ast.Tuple)):
        raise RuntimeError(
            "Unexpected AST node for registration entries: "
            f"{tuple_node.__class__.__name__}"
        )

    pairs: List[Tuple[str, str]] = []
    for entry in tuple_node.elts:  # type: ignore[attr-defined]
        if not isinstance(entry, (ast.Tuple, ast.List)) or len(entry.elts) < 1:
            continue
        key_node = entry.elts[0]
        target_node = entry.elts[1] if len(entry.elts) > 1 else None

        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            key = key_node.value
        else:
            # Skip entries where the key is not a constant string.
            continue

        if isinstance(target_node, ast.Name):
            target_repr = target_node.id
        elif isinstance(target_node, ast.Attribute):
            target_repr = ast.unparse(target_node)  # type: ignore[attr-defined]
        else:
            target_repr = "<unknown>"

        pairs.append((key, target_repr))

    return pairs


_NORMALIZE_OUT_SUFFIX = re.compile(r"_out$", re.IGNORECASE)
_NORMALIZE_BACKWARD_SUFFIX = re.compile(r"_backward$", re.IGNORECASE)
SPECIAL_NORMALIZE_MAP: Dict[str, str] = {
    # some alias here
    "divide": "div",
    "divide_": "div_",
    "true_divide": "div",
    "true_divide_": "div_",
    # special backward ops
    "log_softmax_backward_data": "log_softmax",
    "softmax_backward_data": "softmax",
    # forward ops
    "nll_loss2d_forward": "nll_loss2d",
    "nll_loss_forward": "nll_loss",
    # native ops
    "native_batch_norm": "batch_norm",
    "native_dropout": "dropout",
    "native_group_norm": "group_norm",
    "native_layer_norm": "layer_norm",
    # linalg ops
    "linalg_vector_norm": "vector_norm",
    # other special ops
    "max_pool2d_with_indices": "max_pool2d",
    "moe_align_block_size_triton": "moe_align_block_size",
    "unique2": "unique",
    "weight_norm_interface": "weight_norm",
    "constant_pad_nd": "pad",
}

SPECIAL_CATEGORY_SOURCES: Dict[str, Dict[str, Set[str]]] = {
    "nn.functional": {
        "conv1d": {"torch.nn.functional.conv1d"},
        "conv2d": {"torch.nn.functional.conv2d"},
        "conv3d": {"torch.nn.functional.conv3d"},
        "scaled_dot_product_attention": {
            "torch.nn.functional.scaled_dot_product_attention"
        },
    },
    "vllm": {
        "flash_attn_varlen_func": {"vllm.flash_attn_varlen_func"},
        "get_scheduler_metadata": {"vllm.get_scheduler_metadata"},
    },
}


def normalize_key(key: str) -> str:
    """Normalize an aten key according to heuristic rules.

    Current rules (extend as needed):
    * drop any dotted suffix, e.g. ``add.Tensor`` -> ``add``
    * drop a trailing ``_out`` suffix, treating ``mm_out`` as ``mm``
    * drop a trailing ``_backward`` suffix, treating ``foo_backward`` as ``foo``
    * strip leading underscores so private operators like ``_foo`` normalize to ``foo``
    * remap selected canonical names via :data:`SPECIAL_NORMALIZE_MAP`
    """

    base = key.split(".", 1)[0]
    base = _NORMALIZE_OUT_SUFFIX.sub("", base)
    base = _NORMALIZE_BACKWARD_SUFFIX.sub("", base)
    base = base.lstrip("_")
    return SPECIAL_NORMALIZE_MAP.get(base, base)


def extract_ops(
    init_path: Path = DEFAULT_INIT_PATH,
    *,
    normalized: bool = True,
) -> Dict[str, Dict[str, Set[str]]]:
    pairs = _collect_registration_pairs(init_path)
    fused_exports = _collect_fused_exports()

    def transform(name: str) -> str:
        return normalize_key(name) if normalized else name

    categories: Dict[str, Dict[str, Set[str]]] = {
        "aten": defaultdict(set),
        "fused": defaultdict(set),
    }

    for key, _ in pairs:
        categories["aten"][transform(key)].add(key)

    for key in fused_exports:
        categories["fused"][transform(key)].add(key)

    for label, mapping in SPECIAL_CATEGORY_SOURCES.items():
        cat_map: Dict[str, Set[str]] = defaultdict(set)
        for canonical, raw_values in mapping.items():
            entry_key = transform(canonical)
            cat_map[entry_key].add(canonical)
            for raw in raw_values:
                cat_map[entry_key].add(raw)
        categories[label] = cat_map

    def _freeze(map_: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        return {name: set(raws) for name, raws in map_.items()}

    return {label: _freeze(map_) for label, map_ in categories.items()}


def _collect_fused_exports(path: Path = FUSED_INIT_PATH) -> Set[str]:
    if not path.exists():
        return set()

    module_ast = ast.parse(path.read_text())
    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    exports: Set[str] = set()
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            exports.add(elt.value)
                    return exports
    return set()


def _build_entries(
    category_maps: Dict[str, Dict[str, Set[str]]],
) -> List[Tuple[str, List[str], str]]:
    entries: List[Tuple[str, List[str], str]] = []
    for source, mapping in category_maps.items():
        for name, raws in mapping.items():
            entries.append((name, sorted(raws), source))
    entries.sort(key=lambda item: (item[0], item[2]))
    return entries


def _format_output(
    category_maps: Dict[str, Dict[str, Set[str]]],
    args: argparse.Namespace,
) -> str:
    entries = _build_entries(category_maps)
    counts = {source: len(mapping) for source, mapping in category_maps.items()}
    unique_names = sorted({name for name, *_ in entries})

    if args.detail == "names":
        if args.output_format == "json":
            payload = {
                "names": unique_names,
                "counts": {**counts, "total": len(unique_names)},
            }
            return json.dumps(payload, indent=2, ensure_ascii=False)
        lines = ["# Operator names:"]
        lines.extend(unique_names)
        return "\n".join(lines)

    if args.output_format == "json":
        payload = {
            "entries": [
                {"name": name, "raw": raw, "source": source}
                for name, raw, source in entries
            ],
            "counts": {**counts, "total": len(unique_names)},
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    lines: List[str] = ["# Operator tuples (name, raw_list, source):"]
    for name, raw, source in entries:
        lines.append(f"({name!r}, {raw}, {source!r})")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-path",
        type=Path,
        default=DEFAULT_INIT_PATH,
        help="Path to the flag_gems __init__.py file (auto-detected by default)",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalized",
        action="store_false",
        help="Skip normalization and only return the raw keys.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--detail",
        choices=("names", "full"),
        default="names",
        help="names: only operator names(default); full: emit full tuples.",
    )
    args = parser.parse_args(argv)

    category_maps = extract_ops(
        init_path=args.init_path,
        normalized=args.normalized,
    )
    output = _format_output(category_maps, args)
    print(output)


if __name__ == "__main__":
    main()
