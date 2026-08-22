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

"""Declarative op specifications for the C++ pointwise dispatch glue.

This module is the *single source of truth* for the aten-facing C++ glue that
wraps the generated ``pointwise_dynamic::<fn>_func`` kernels. From these specs,
``prebuild_kernels.py`` generates, at build time:

  - ``pointwise_ops_glue.h``  : ``flag_gems::<op>(...)`` declarations
                                (included by ``flag_gems/operators.h``)
  - ``pointwise_ops_glue.cc`` : function bodies for *passthrough* / *selector*
                                ops (unary elementwise + gelu). Ops whose body
                                is genuinely custom (add/div/fill/remainder --
                                complex dispatch, rounding modes, Scalar->double,
                                0-dim branching) set ``handwritten=True`` and
                                keep their body in ``lib/<op>.cpp``.
  - ``pointwise_cstub.inc``   : the three registration blocks
                                (pybind ``m.def`` / ``TORCH_LIBRARY`` schema /
                                ``TORCH_LIBRARY_IMPL`` ``m.impl``) included by
                                ``csrc/cstub.cpp``.

Adding a new elementwise op is therefore a single ``passthrough(...)`` (or
``selector(...)``) entry here plus its module in the CMake op-file list -- no
hand-written C++ across four files.

The kernel *structure* (arity, promotion, per-rank kernel symbols) is still
discovered automatically from the ``@pointwise_dynamic`` decorator by
``prebuild_kernels.py``; the specs here only add the aten-facing information the
decorator does not carry (public schema, kwarg names/defaults, and the
aten-name -> kernel-symbol mapping incl. value-dependent selection).
"""

from dataclasses import dataclass
from typing import List, Optional

# ===================================================================
# Parameter model
# ===================================================================

# Param kinds -> (C++ type, is_tensor). ``mut`` variants are non-const refs
# used by in-place / out ops.
_PARAM_TYPES = {
    "tensor": ("const at::Tensor &", True),
    "tensor_mut": ("at::Tensor &", True),
    "scalar": ("const at::Scalar &", False),  # at::Scalar by const-ref
    "cscalar": ("const c10::Scalar &", False),  # c10::Scalar spelling (fill)
    "double": ("double", False),
    "str": ("c10::string_view", False),
    "optstr": ("const c10::optional<std::string> &", False),
}


@dataclass
class Param:
    """One C++ parameter of a ``flag_gems::<op>`` function.

    Attributes:
        kind: One of the keys in ``_PARAM_TYPES``.
        name: Parameter name as it appears in the signature.
        default: C++ default-value literal (e.g. ``"1"``, ``'"none"'``), or
            ``None`` for a required parameter.
    """

    kind: str
    name: str
    default: Optional[str] = None

    @property
    def cpp_type(self) -> str:
        return _PARAM_TYPES[self.kind][0]

    @property
    def is_tensor(self) -> bool:
        return _PARAM_TYPES[self.kind][1]

    def decl(self, with_default: bool) -> str:
        """Render as ``<type> <name>[ = <default>]`` for a declaration."""
        t = self.cpp_type
        sep = "" if t.endswith("&") else " "
        s = f"{t}{sep}{self.name}"
        if with_default and self.default is not None:
            s += f" = {self.default}"
        return s

    def defn(self) -> str:
        """Render for a definition (no default value)."""
        return self.decl(with_default=False)


@dataclass
class OpSpec:
    """One aten-facing C++ glue function + its registration.

    A spec produces exactly one ``operators.h`` declaration and (unless
    ``handwritten``) one ``pointwise_ops_glue.cc`` body. Registration into the
    three cstub blocks is controlled by ``pybind`` / ``schema`` / ``impl_name``.
    """

    cpp_name: str  # flag_gems::<cpp_name>
    params: List[Param]
    ret: str = "at::Tensor"  # or "at::Tensor &"
    decl_comment: Optional[str] = None  # e.g. "abs(Tensor self) -> Tensor"
    decl_override: Optional[str] = None  # full custom declaration text (rare)

    # --- body generation strategy (ignored when handwritten) ---
    kernel: Optional[str] = None  # passthrough: pointwise_dynamic::<kernel>(...)
    select: Optional[dict] = (
        None  # {"on": <param>, "cases": {...}, "default": <kernel>,
    )
    #                                 "check_msg": <str>}
    handwritten: bool = False  # body lives in a hand-written lib/<op>.cpp

    # --- boxed dispatch routing (used only by the boxed adapter, see
    #     pointwise_boxed.h). Independent of `handwritten`: even hand-written
    #     bodies (add/div/fill) route through the generic boxed adapter, so
    #     they set `kernel` here too. ---
    # kernel_ts / kernel_st: the tensor-scalar / scalar-tensor kernel variants
    # for a *binary* op. Required because a 0-dim tensor operand crashes the
    # base rank-N kernel (Triton block-ptr on a scalar) -- the adapter must
    # extract the 0-dim operand as a scalar and route to the matching variant.
    # remainder breaks the "<kernel>_tensor_scalar" suffix convention
    # (rem_tt/rem_ts/rem_st), so the variants are named explicitly rather than
    # derived by suffix.
    kernel_ts: Optional[str] = None  # binary: 2nd operand 0-dim/scalar
    kernel_st: Optional[str] = None  # binary: 1st operand 0-dim/scalar
    # boxed_select: value-dependent kernel choice inside the adapter, keyed on
    # a string arg. {"on": "rounding_mode", "cases": {"floor": "floor_div_func",
    # "trunc": "trunc_div_func"}, "default": "true_div_func", "family": True}
    # ("family": True means each chosen kernel is itself a binary family, so the
    # 0-dim rule still applies with <kernel>{,_tensor_scalar,_scalar_tensor}).
    boxed_select: Optional[dict] = None
    # boxed_both0_host: remainder-only. When BOTH operands are 0-dim tensors the
    # hand-written lib/div.cpp computes the result on the host (std::fmod with a
    # sign fix) instead of launching a kernel; the adapter replicates that.
    boxed_both0_host: bool = False

    # --- cstub registration ---
    # pybind: None => skip; "name" => m.def("name", &flag_gems::<cpp_name>);
    # dict   => lambda form with py::arg defaults (see _gen_pybind).
    pybind: object = None
    schema: Optional[str] = None  # TORCH_LIBRARY m.def string; None => impl-only
    schema_tags: Optional[str] = None  # extra 2nd arg to schema m.def (e.g. tag set)
    impl_name: Optional[str] = None  # TORCH_LIBRARY_IMPL m.impl aten name

    # cross-check: expected (num_input_tensors, num_non_tensor_inputs) against
    # the discovered @pointwise_dynamic FunctionSchema for ``kernel``. Skipped
    # when kernel is None (selector/handwritten multi-kernel).
    check_arity: Optional[tuple] = None


@dataclass
class ImplOnly:
    """An extra ``m.impl`` registration for a separately declared schema.

    Custom ``flag_gems`` overloads must have local schemas because the boxed
    adapter queries ``op.schema()`` before decoding the boxed stack.
    """

    aten_name: str
    cpp_name: str


# ===================================================================
# Authoring helpers
# ===================================================================


def passthrough(cpp_name: str, aten_schema: str, kernel: str) -> OpSpec:
    """A unary op that forwards directly to a single kernel.

    ``aten_schema`` is the full schema string (e.g. ``"abs(Tensor self) -> Tensor"``);
    its head (before ``(``) is used as the pybind/impl name.
    """
    aten_name = aten_schema.split("(", 1)[0]
    return OpSpec(
        cpp_name=cpp_name,
        params=[Param("tensor", "self")],
        ret="at::Tensor",
        decl_comment=aten_schema,
        kernel=kernel,
        pybind=aten_name,
        schema=aten_schema,
        impl_name=aten_name,
        check_arity=(1, 0),
    )


# ===================================================================
# The specs
# ===================================================================
# NOTE: order here defines emission order in the generated files, which is
# matched against the current hand-written glue for byte-alignment.

# ---- add family (bodies hand-written in lib/add.cpp) ----
_ADD: List[OpSpec] = [
    OpSpec(
        cpp_name="add_tensor",
        params=[
            Param("tensor", "self"),
            Param("tensor", "other"),
            Param("scalar", "alpha", "1"),
        ],
        decl_comment="add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
        handwritten=True,
        # Plain boxed path: add never promotes a 0-dim operand to scalar (the
        # base add_func handles any-rank b, and its tensor_scalar variant carries
        # *two* trailing scalars (other, alpha) which the binary 0-dim swap can't
        # express). Matches the un-guarded hand-written lib/add.cpp exactly.
        kernel="add_func",
        pybind={
            "name": "add_tensor",
            "params": [
                ("const at::Tensor& self", "self", None),
                ("const at::Tensor& other", "other", None),
                ("double alpha", "alpha", "1.0"),
            ],
        },
        schema="add_tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
        schema_tags="{at::Tag::pt2_compliant_tag}",
        impl_name="add_tensor",
    ),
    OpSpec(
        cpp_name="add_scalar",
        params=[
            Param("tensor", "self"),
            Param("scalar", "other"),
            Param("scalar", "alpha", "1"),
        ],
        decl_comment="add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
        handwritten=True,
        kernel="add_func_tensor_scalar",
        pybind={
            "name": "add_scalar",
            "params": [
                ("const at::Tensor& self", "self", None),
                ("const at::Scalar& other", "other", None),
                ("double alpha", "alpha", "1.0"),
            ],
        },
        schema="add_scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor",
        schema_tags="{at::Tag::pt2_compliant_tag}",
        impl_name="add_scalar",
    ),
    OpSpec(
        cpp_name="add_tensor_inplace",
        params=[
            Param("tensor_mut", "self"),
            Param("tensor", "other"),
            Param("scalar", "alpha", "1"),
        ],
        ret="at::Tensor &",
        decl_comment="add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
        handwritten=True,
        kernel="add_func",  # Plain path (see add_tensor note).
        pybind={
            "name": "add_tensor_inplace",
            "params": [
                ("at::Tensor& self", "self", None),
                ("const at::Tensor& other", "other", None),
                ("double alpha", "alpha", "1.0"),
            ],
        },
        schema="add_tensor_inplace(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
        schema_tags="{at::Tag::pt2_compliant_tag}",
        impl_name="add_tensor_inplace",
    ),
    OpSpec(
        cpp_name="add_scalar_inplace",
        params=[
            Param("tensor_mut", "self"),
            Param("scalar", "other"),
            Param("scalar", "alpha", "1"),
        ],
        ret="at::Tensor &",
        decl_comment="add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
        handwritten=True,
        kernel="add_func_tensor_scalar",
        pybind={
            "name": "add_scalar_inplace",
            "params": [
                ("at::Tensor& self", "self", None),
                ("const at::Scalar& other", "other", None),
                ("double alpha", "alpha", "1.0"),
            ],
        },
        schema="add_scalar_inplace(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)",
        schema_tags="{at::Tag::pt2_compliant_tag}",
        impl_name="add_scalar_inplace",
    ),
]

# ---- div / remainder family (bodies hand-written in lib/div.cpp) ----
# Declarations only (no per-decl comment in the current header); a blank line
# precedes this group. div_mode / div_mode_ have multi-line decls preserved via
# decl_override.
_DIV: List[OpSpec] = [
    OpSpec(
        "true_div",
        [Param("tensor", "a"), Param("tensor", "b")],
        handwritten=True,
        kernel="true_div_func",
        kernel_ts="true_div_func_tensor_scalar",
        kernel_st="true_div_func_scalar_tensor",
    ),
    OpSpec(
        "true_div_",
        [Param("tensor_mut", "a"), Param("tensor", "b")],
        ret="at::Tensor",
        handwritten=True,
        kernel="true_div_func",
        kernel_ts="true_div_func_tensor_scalar",
    ),
    OpSpec(
        "trunc_div",
        [Param("tensor", "a"), Param("tensor", "b")],
        handwritten=True,
        kernel="trunc_div_func",
        kernel_ts="trunc_div_func_tensor_scalar",
        kernel_st="trunc_div_func_scalar_tensor",
    ),
    OpSpec(
        "trunc_div_",
        [Param("tensor_mut", "a"), Param("tensor", "b")],
        ret="at::Tensor",
        handwritten=True,
        kernel="trunc_div_func",
        kernel_ts="trunc_div_func_tensor_scalar",
    ),
    OpSpec(
        "floor_div",
        [Param("tensor", "a"), Param("tensor", "b")],
        handwritten=True,
        kernel="floor_div_func",
        kernel_ts="floor_div_func_tensor_scalar",
        kernel_st="floor_div_func_scalar_tensor",
    ),
    OpSpec(
        "floor_div_",
        [Param("tensor_mut", "a"), Param("tensor", "b")],
        ret="at::Tensor",
        handwritten=True,
        kernel="floor_div_func",
        kernel_ts="floor_div_func_tensor_scalar",
    ),
    OpSpec(
        "div_mode",
        [Param("tensor", "a"), Param("tensor", "b"), Param("optstr", "rounding_mode")],
        handwritten=True,
        boxed_select={
            "on": "rounding_mode",
            "cases": {
                "floor": "floor_div_func",
                "trunc": "trunc_div_func",
                "none": "true_div_func",
            },
            "default": "true_div_func",
            "family": True,
        },
    ),
    OpSpec(
        "div_mode_",
        [
            Param("tensor_mut", "a"),
            Param("tensor", "b"),
            Param("optstr", "rounding_mode"),
        ],
        ret="at::Tensor",
        handwritten=True,
        boxed_select={
            "on": "rounding_mode",
            "cases": {
                "floor": "floor_div_func",
                "trunc": "trunc_div_func",
                "none": "true_div_func",
            },
            "default": "true_div_func",
            "family": True,
        },
    ),
    OpSpec(
        "remainder_tt",
        [Param("tensor", "a"), Param("tensor", "b")],
        handwritten=True,
        kernel="rem_tt",
        kernel_ts="rem_ts",
        kernel_st="rem_st",
    ),
    OpSpec(
        "remainder_ts",
        [Param("tensor", "a"), Param("double", "b_scalar")],
        handwritten=True,
        kernel="rem_ts",
    ),
    OpSpec(
        "remainder_st",
        [Param("double", "a_scalar"), Param("tensor", "b")],
        handwritten=True,
        kernel="rem_st",
    ),
    OpSpec(
        "remainder",
        [Param("tensor", "a"), Param("tensor", "b")],
        handwritten=True,
        kernel="rem_tt",
        kernel_ts="rem_ts",
        kernel_st="rem_st",
        boxed_both0_host=True,
    ),
    OpSpec(
        "remainder_",
        [Param("tensor_mut", "a"), Param("tensor", "b")],
        ret="at::Tensor",
        handwritten=True,
        kernel="rem_tt",
        kernel_ts="rem_ts",
    ),
]

# ---- fill family (bodies hand-written in lib/fill.cpp) ----
_FILL: List[OpSpec] = [
    OpSpec(
        "fill_scalar",
        [Param("tensor", "input"), Param("cscalar", "value")],
        handwritten=True,
        kernel="fill_scalar_func",
    ),
    OpSpec(
        "fill_tensor",
        [Param("tensor", "input"), Param("tensor", "value")],
        handwritten=True,
        kernel="fill_tensor_func",
    ),
    OpSpec(
        "fill_scalar_",
        [Param("tensor_mut", "input"), Param("cscalar", "value")],
        ret="at::Tensor &",
        handwritten=True,
        kernel="fill_scalar_func",
    ),
    OpSpec(
        "fill_tensor_",
        [Param("tensor_mut", "input"), Param("tensor", "value")],
        ret="at::Tensor &",
        handwritten=True,
        kernel="fill_tensor_func",
    ),
]

# ---- unary elementwise (generated bodies -> pointwise_ops_glue.cc) ----
_UNARY: List[OpSpec] = [
    passthrough("abs", "abs(Tensor self) -> Tensor", "abs_func"),
    passthrough("neg", "neg(Tensor self) -> Tensor", "neg_func"),
    passthrough("exp", "exp(Tensor self) -> Tensor", "exp_func"),
    passthrough("sqrt", "sqrt(Tensor self) -> Tensor", "sqrt_func"),
    passthrough("rsqrt", "rsqrt(Tensor self) -> Tensor", "rsqrt_func"),
    passthrough("tanh", "tanh(Tensor self) -> Tensor", "tanh_kernel"),
    passthrough("sigmoid", "sigmoid(Tensor self) -> Tensor", "sigmoid_forward"),
    passthrough("silu", "silu(Tensor self) -> Tensor", "silu_forward"),
    passthrough("relu", "relu(Tensor self) -> Tensor", "relu_forward"),
]

# ---- gelu (selector body -> pointwise_ops_glue.cc) ----
_GELU = OpSpec(
    cpp_name="gelu",
    params=[Param("tensor", "self"), Param("str", "approximate", '"none"')],
    decl_comment="gelu(Tensor self, *, str approximate='none') -> Tensor",
    select={
        "on": "approximate",
        "cases": {"tanh": "gelu_tanh"},
        "default": "gelu_none",
        "check": ("none", "gelu: approximate must be 'none' or 'tanh', got '"),
    },
    boxed_select={
        "on": "approximate",
        "cases": {"tanh": "gelu_tanh"},
        "default": "gelu_none",
        "family": False,
    },
    pybind={
        "name": "gelu",
        "params": [
            ("const at::Tensor& self", "self", None),
            ("const std::string& approximate", "approximate", '"none"'),
        ],
    },
    schema="gelu(Tensor self, *, str approximate='none') -> Tensor",
    impl_name="gelu",
)

# ------------------------------------------------------------------
# Legacy div/fill/remainder cstub registrations.
#
# These families map many dotted aten overloads onto a few hand-written C++
# fns (true_div/floor_div/div_mode/remainder/...). Their registration order and
# fan-out is irregular, so we encode the three cstub blocks as faithful ordered
# lists rather than deriving them from cpp_name. Each entry: (aten_name, cpp_fn).
# ------------------------------------------------------------------

# pybind ``m.def`` block (mirrors cstub.cpp "div" + "fill" groups).
_DIV_PYBIND = [
    ("div.Tensor", "true_div"),
    ("div_.Tensor", "true_div_"),
    ("div.Tensor_mode", "div_mode"),
    ("div_.Tensor_mode", "div_mode_"),
    ("floor_divide", "floor_div"),
    ("floor_divide_.Tensor", "floor_div_"),
    ("divide.Tensor", "true_div"),
    ("divide_.Tensor", "true_div_"),
    ("divide.Tensor_mode", "div_mode"),
    ("divide_.Tensor_mode", "div_mode_"),
    ("true_divide.Tensor", "true_div"),
    ("true_divide_.Tensor", "true_div_"),
    ("remainder.Tensor", "remainder"),
    ("remainder_.Tensor", "remainder_"),
]
_FILL_PYBIND = [
    ("fill.Scalar", "fill_scalar"),
    ("fill.Tensor", "fill_tensor"),
    ("fill_.Scalar", "fill_scalar_"),
    ("fill_.Tensor", "fill_tensor_"),
]

# TORCH_LIBRARY schema block (schema strings, in file order).
# Must be 1:1 aligned with _DIV_IMPL — every impl needs a schema in the custom
# flag_gems namespace because the boxed adapter calls op.schema().
_DIV_SCHEMA = [
    "div.Tensor(Tensor self, Tensor other) -> Tensor",
    "div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
    "div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor",
    "div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)",
    "div.Scalar(Tensor self, Scalar other) -> Tensor",
    "div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
    "div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor",
    "div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)",
    "floor_divide(Tensor self, Tensor other) -> Tensor",
    "floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
    "floor_divide.Scalar(Tensor self, Scalar other) -> Tensor",
    "floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
    "divide.Tensor(Tensor self, Tensor other) -> Tensor",
    "divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
    "divide.Scalar(Tensor self, Scalar other) -> Tensor",
    "divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
    "divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor",
    "divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)",
    "divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor",
    "divide_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)",
    "true_divide.Tensor(Tensor self, Tensor other) -> Tensor",
    "true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
    "remainder.Scalar(Tensor self, Scalar other) -> Tensor",
    "remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)",
    "remainder.Tensor(Tensor self, Tensor other) -> Tensor",
    "remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)",
    "remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor",
]
_FILL_SCHEMA = [
    "fill.Scalar(Tensor self, Scalar value) -> Tensor",
    "fill.Tensor(Tensor self, Tensor value) -> Tensor",
    "fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)",
    "fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)",
]

# TORCH_LIBRARY_IMPL block (complete, faithful order incl. Scalar overloads
# that have no local schema -- their aten schema comes from PyTorch built-ins).
_DIV_IMPL = [
    ("div.Tensor", "true_div"),
    ("div_.Tensor", "true_div_"),
    ("div.Tensor_mode", "div_mode"),
    ("div_.Tensor_mode", "div_mode_"),
    ("div.Scalar", "true_div"),
    ("div_.Scalar", "true_div_"),
    ("div.Scalar_mode", "div_mode"),
    ("div_.Scalar_mode", "div_mode_"),
    ("floor_divide", "floor_div"),
    ("floor_divide_.Tensor", "floor_div_"),
    ("floor_divide.Scalar", "floor_div"),
    ("floor_divide_.Scalar", "floor_div_"),
    ("divide.Tensor", "true_div"),
    ("divide_.Tensor", "true_div_"),
    ("divide.Scalar", "true_div"),
    ("divide_.Scalar", "true_div_"),
    ("divide.Tensor_mode", "div_mode"),
    ("divide_.Tensor_mode", "div_mode_"),
    ("divide.Scalar_mode", "div_mode"),
    ("divide_.Scalar_mode", "div_mode_"),
    ("true_divide.Tensor", "true_div"),
    ("true_divide_.Tensor", "true_div_"),
    ("remainder.Scalar", "remainder"),
    ("remainder_.Scalar", "remainder_"),
    ("remainder.Tensor", "remainder"),
    ("remainder_.Tensor", "remainder_"),
    ("remainder.Scalar_Tensor", "remainder"),
]
_FILL_IMPL = [
    ("fill.Scalar", "fill_scalar"),
    ("fill.Tensor", "fill_tensor"),
    ("fill_.Scalar", "fill_scalar_"),
    ("fill_.Tensor", "fill_tensor_"),
]

# All declaration-bearing specs in header/body emission order.
ALL_SPECS: List[OpSpec] = _ADD + _DIV + _FILL + _UNARY + [_GELU]

# Legacy family registration tables, keyed by cstub block.
LEGACY_PYBIND = _DIV_PYBIND + _FILL_PYBIND
LEGACY_SCHEMA = _DIV_SCHEMA + _FILL_SCHEMA
LEGACY_IMPL = _DIV_IMPL + _FILL_IMPL


def specs_needing_body() -> List[OpSpec]:
    """Specs whose body is generated (passthrough/selector, not hand-written)."""
    return [s for s in ALL_SPECS if not s.handwritten]
