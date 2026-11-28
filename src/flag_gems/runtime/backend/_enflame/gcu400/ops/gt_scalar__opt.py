"""gt_scalar_ thin re-export shim (2026-11-16 perf round).

aten::gt_.Scalar used to dispatch here from a historical candidate variant
(one-block-fits-all kernel with a fp32 up-cast, which both wastes ALU on
large tiers and compares fp16/bf16 inputs against the fp32 representation
of the scalar instead of casting the scalar to ``A.dtype`` like torch's
TensorIterator). The tiered implementation in ``.gt`` already provides the
correct and faster in-place scalar path (single-CTA tiny / flat / grid-stride
cap-48 kernels, native-dtype comparison), so this module just re-exports it
and keeps the import path in ``ops/__init__.py`` stable.
"""

from .gt import gt_scalar_

__all__ = ["gt_scalar_"]
