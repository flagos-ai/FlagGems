// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// ==========================================================================
// Generic boxed dispatch adapter for FlagGems pointwise ops.
//
// A SINGLE function -- flaggems_pointwise_boxed -- is registered (via
// torch::CppFunction::makeFromBoxedFunction) for EVERY pointwise aten op when
// the build is configured with FLAGGEMS_POINTWISE_DYNAMIC_BOXED. All per-op
// behaviour is data-driven by the generated ATEN_TO_BOXED table
// (pointwise_aten_to_kernel.h). Adding a new pointwise op then costs one line
// in op_specs.py and zero hand-written C++.
//
// This replaces the per-op TORCH_FN glue (pointwise_ops_glue.cc + the
// hand-written lib/{add,div,fill}.cpp bodies). The routing here mirrors those
// bodies exactly:
//   * Plain        -- unary / add / fill: tensors are kernel inputs in order,
//                     scalars become double scalar_args in order, mask follows
//                     the natural argument order.
//   * Binary       -- div / floor / trunc / remainder: a 0-dim tensor operand
//                     MUST be promoted to a scalar (the rank-N Triton kernel
//                     block-ptrs a scalar base and crashes otherwise), routing
//                     to the tensor_scalar / scalar_tensor kernel variant.
//   * SelectPlain  -- gelu: a string arg picks the kernel, then Plain.
//   * SelectBinary -- div_mode: a string arg picks the family base, then Binary
//                     (variant names derived by _tensor_scalar/_scalar_tensor
//                     suffix on the chosen base).
// ==========================================================================

// NOTE: this is a DECLARATION-only header. The definition lives in
// pointwise_boxed.cc, which is compiled as part of the `operators` library
// (the only target with the CUDA / triton_jit include paths that
// pointwise_prepare_args.h transitively needs). cstub.cpp includes THIS header
// merely to name the function in makeFromBoxedFunction<&...>(); pulling the
// full definition (and its CUDA headers) into the c_operators pybind target
// would fail to find cuda_runtime_api.h.

#include <ATen/core/dispatch/OperatorEntry.h>
#include <ATen/core/stack.h>

namespace flag_gems {

// The one boxed kernel registered for every pointwise op (defined in
// pointwise_boxed.cc). All per-op behaviour is data-driven by the generated
// ATEN_TO_BOXED recipe table.
void flaggems_pointwise_boxed(const c10::OperatorHandle& op, torch::jit::Stack* stack);

}  // namespace flag_gems

#ifdef FLAGGEMS_POINTWISE_BOXED_IMPL
// ==========================================================================
// Definition (compiled only into `operators` via pointwise_boxed.cc, which
// #defines FLAGGEMS_POINTWISE_BOXED_IMPL before including this header).
// ==========================================================================

#include <c10/util/Optional.h>
#include <torch/library.h>

#include <cmath>
#include <string>
#include <vector>

#include "pointwise_aten_to_kernel.h"  // generated: BoxedRecipe + ATEN_TO_BOXED
#include "pointwise_prepare_args.h"    // pointwise_dynamic::dispatch_pointwise

namespace flag_gems {

namespace detail {

  // One decoded positional argument off the boxed stack.
  struct BoxedArg {
    bool is_tensor = false;
    at::Tensor tensor;    // valid when is_tensor
    double scalar = 0.0;  // valid when !is_tensor
    at::ScalarType scalar_dtype = at::kDouble;  // valid when !is_tensor
  };

  // Derive the "<base>_tensor_scalar" / "<base>_scalar_tensor" variant name for a
  // binary family kernel. remainder breaks this convention (rem_tt/rem_ts/rem_st)
  // so its variants are carried explicitly in the recipe and this is never used
  // for it; div/floor/trunc all follow the suffix rule.
  inline std::string variant_name(const std::string& base, const char* suffix) {
    return base + suffix;
  }

  // Dispatch a binary op given already-classified operands `a` and `b`, applying
  // the 0-dim -> scalar promotion rule. `base`/`ts`/`st` are the three kernel
  // keys (ts/st may be empty for in-place ops whose 1st operand can't be 0-dim).
  // `pre_out` (optional) forces an output tensor for the in-place path.
  inline at::Tensor dispatch_binary(const std::string& base,
                                    const std::string& ts,
                                    const std::string& st,
                                    const BoxedArg& a,
                                    const BoxedArg& b,
                                    const c10::optional<at::Tensor>& pre_out) {
    namespace pd = pointwise_dynamic;
    std::vector<c10::optional<at::Tensor>> outs;
    if (pre_out.has_value()) outs = {pre_out};

    // scalar operand(s) supplied directly by the aten overload (e.g. div.Scalar):
    // route to the matching variant without touching tensor ranks.
    // Kernel arg order follows the mask positionally (pointwise_prepare_args.h:
    // build kernel arguments). The *_tensor_scalar kernel is compiled as
    // is_tensor=[True, False] (tensor@0, scalar@1) -> mask {true,false}; the
    // *_scalar_tensor kernel is is_tensor=[False, True] (scalar@0, tensor@1) ->
    // mask {false,true}. Passing the wrong order block-ptrs a scalar and crashes
    // Triton ("Expected base to be a pointer type").
    if (a.is_tensor && !b.is_tensor) {
      TORCH_CHECK(!ts.empty(), "flaggems boxed: missing tensor_scalar variant");
      return pd::dispatch_pointwise(
          ts, {a.tensor}, {b.scalar}, {b.scalar_dtype}, {true, false}, outs);
    }
    if (!a.is_tensor && b.is_tensor) {
      TORCH_CHECK(!st.empty(), "flaggems boxed: missing scalar_tensor variant");
      return pd::dispatch_pointwise(
          st, {b.tensor}, {a.scalar}, {a.scalar_dtype}, {false, true}, outs);
    }

    // both tensors: apply the 0-dim promotion rule (mirrors lib/div.cpp).
    const at::Tensor& ta = a.tensor;
    const at::Tensor& tb = b.tensor;
    if (ta.dim() == 0 && tb.dim() > 0) {
      TORCH_CHECK(!st.empty(), "flaggems boxed: missing scalar_tensor variant");
      return pd::dispatch_pointwise(
          st,
          {tb},
          {ta.item<double>()},
          {ta.scalar_type()},
          {false, true},
          outs);
    }
    if (tb.dim() == 0) {
      TORCH_CHECK(!ts.empty(), "flaggems boxed: missing tensor_scalar variant");
      return pd::dispatch_pointwise(
          ts,
          {ta},
          {tb.item<double>()},
          {tb.scalar_type()},
          {true, false},
          outs);
    }
    return pd::dispatch_pointwise(base, {ta, tb}, {}, {}, {true, true}, outs);
  }

  // remainder's both-0-dim host special case (mirrors lib/div.cpp::remainder).
  inline at::Tensor remainder_both0_host(const at::Tensor& a, const at::Tensor& b) {
    double av = a.item<double>();
    double bv = b.item<double>();
    double r = std::fmod(av, bv);
    if (r != 0.0 && ((r < 0.0) != (bv < 0.0))) r += bv;
    return torch::tensor(r, a.options());
  }

}  // namespace detail

// The one boxed kernel registered for every pointwise op.
void flaggems_pointwise_boxed(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const c10::FunctionSchema& schema = op.schema();
  const std::string& op_name = schema.name();  // e.g. "flag_gems::div"

  // The registered key includes the overload (e.g. "div.Tensor"); reconstruct
  // "flag_gems::<name>.<overload>" to match the ATEN_TO_BOXED table.
  std::string key = op_name;
  const std::string& overload = schema.overload_name();
  if (!overload.empty()) key += "." + overload;

  auto it = ATEN_TO_BOXED.find(key);
  TORCH_CHECK(it != ATEN_TO_BOXED.end(), "flaggems_pointwise_boxed: no routing recipe for '", key, "'");
  const BoxedRecipe& recipe = it->second;

  const int nargs = static_cast<int>(schema.arguments().size());

  // Decode every positional arg off the stack, and note whether arg 0 is a
  // write-alias (in-place op) so we can force it as the output tensor.
  std::vector<detail::BoxedArg> args;
  args.reserve(nargs);
  std::string str_arg;  // the (single) string arg, for Select* kinds
  bool have_str = false;
  bool inplace = false;
  for (int i = 0; i < nargs; ++i) {
    const c10::IValue& iv = torch::jit::peek(*stack, i, nargs);
    detail::BoxedArg a;
    if (iv.isTensor()) {
      a.is_tensor = true;
      a.tensor = iv.toTensor();
      if (i == 0) {
        const auto& alias = schema.arguments()[i].alias_info();
        inplace = alias && alias->isWrite();
      }
    } else if (iv.isScalar()) {
      a.scalar = iv.toScalar().toDouble();
      a.scalar_dtype = iv.toScalar().type();
    } else if (iv.isInt()) {
      a.scalar = static_cast<double>(iv.toInt());
      a.scalar_dtype = at::kLong;
    } else if (iv.isDouble()) {
      a.scalar = iv.toDouble();
      a.scalar_dtype = at::kDouble;
    } else if (iv.isBool()) {
      a.scalar = iv.toBool() ? 1.0 : 0.0;
      a.scalar_dtype = at::kBool;
    } else if (iv.isString()) {
      str_arg = iv.toStringRef();
      have_str = true;
      continue;  // strings are selectors, not kernel operands
    } else if (iv.isNone()) {
      continue;  // optional-not-present (e.g. rounding_mode=None)
    } else {
      TORCH_CHECK(false, "flaggems_pointwise_boxed: unsupported arg #", i, " for '", key, "'");
    }
    args.push_back(std::move(a));
  }
  torch::jit::drop(*stack, nargs);

  c10::optional<at::Tensor> pre_out;
  if (inplace && !args.empty() && args[0].is_tensor) pre_out = args[0].tensor;

  // Resolve the base kernel, applying string selectors where present.
  std::string base = recipe.kernel ? recipe.kernel : std::string();
  if (recipe.kind == BoxedKind::SelectPlain || recipe.kind == BoxedKind::SelectBinary) {
    if (have_str) {
      for (const auto& c : recipe.cases) {
        if (str_arg == c.value) {
          base = c.kernel;
          break;
        }
      }
    }
    // Missing selectors use the recipe default (GELU: "none"; division:
    // absent/None/"none" -> true division). An explicitly supplied unknown
    // selector must not silently fall back to a different operation.
    if (have_str) {
      bool matched = false;
      for (const auto& c : recipe.cases) {
        if (str_arg == c.value) {
          matched = true;
          break;
        }
      }
      TORCH_CHECK(
          matched,
          "flaggems_pointwise_boxed: invalid selector '",
          str_arg,
          "' for '",
          key,
          "'");
    }
    TORCH_CHECK(!base.empty(), "flaggems_pointwise_boxed: no base kernel for '", key, "'");
  }

  if ((key == "flag_gems::fill.Tensor" || key == "flag_gems::fill_.Tensor") &&
      args.size() >= 2 && args[1].is_tensor) {
    TORCH_CHECK(
        args[1].tensor.dim() == 0,
        key == "flag_gems::fill.Tensor"
            ? "fill_tensor only supports 0-dim value tensor"
            : "fill_tensor_ only supports 0-dim value tensor");
  }

  at::Tensor result;
  const bool binary = recipe.kind == BoxedKind::Binary || recipe.kind == BoxedKind::SelectBinary;
  if (binary) {
    TORCH_CHECK(args.size() >= 2,
                "flaggems_pointwise_boxed: binary op '",
                key,
                "' expected 2 operands, got ",
                args.size());
    const detail::BoxedArg& a = args[0];
    const detail::BoxedArg& b = args[1];

    if (recipe.both0_host && a.is_tensor && b.is_tensor && a.tensor.dim() == 0 && b.tensor.dim() == 0) {
      result = detail::remainder_both0_host(a.tensor, b.tensor);
    } else {
      // SelectBinary variant names follow the suffix convention; Binary carries
      // explicit ts/st keys (remainder). For SelectBinary derive them; for
      // Binary use the recipe fields.
      std::string ts, st;
      if (recipe.kind == BoxedKind::SelectBinary) {
        ts = detail::variant_name(base, "_tensor_scalar");
        st = detail::variant_name(base, "_scalar_tensor");
        // in-place (div_.*_mode): 1st operand can't be 0-dim -> no st.
        if (inplace) st.clear();
      } else {
        ts = recipe.kernel_ts ? recipe.kernel_ts : std::string();
        st = recipe.kernel_st ? recipe.kernel_st : std::string();
      }
      result = detail::dispatch_binary(base, ts, st, a, b, pre_out);
    }
  } else {
    // Plain / SelectPlain: tensors are inputs (in order), scalars are scalar
    // args (in order); mask follows natural argument order.
    std::vector<at::Tensor> tensors;
    std::vector<double> scalars;
    std::vector<at::ScalarType> scalar_dtypes;
    std::vector<bool> mask;
    tensors.reserve(args.size());
    mask.reserve(args.size());
    for (const auto& a : args) {
      mask.push_back(a.is_tensor);
      if (a.is_tensor) {
        tensors.push_back(a.tensor);
      } else {
        scalars.push_back(a.scalar);
        scalar_dtypes.push_back(a.scalar_dtype);
      }
    }
    std::vector<c10::optional<at::Tensor>> outs;
    if (pre_out.has_value()) outs = {pre_out};
    result = pointwise_dynamic::dispatch_pointwise(
        base, tensors, scalars, scalar_dtypes, mask, outs);
  }

  torch::jit::push(*stack, std::move(result));
}

}  // namespace flag_gems

#endif  // FLAGGEMS_POINTWISE_BOXED_IMPL
