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

// C++ wrapper for ``flag_gems.moe_load_balance_loss`` (Top-K MoE
// load-balancing auxiliary loss, forward-only).
//
// The operator is ``L_ib = N_e * sum_i f_i * P_i``: ``f_i`` is expert ``i``'s
// Top-K assignment frequency and ``P_i`` its mean routing probability, both
// computed in fp32 from the router logits.  The Python side is a two-pass
// Triton pipeline -- a counting pass that histograms the Top-K picks, followed
// by a loss pass that re-reads the logits and dot-products the softmax
// probabilities against the histogram -- because the second pass needs the
// grid-wide histogram produced by the first one.
//
// The Python side is split into a backend-neutral implementation
// (``fused/moe_load_balance_loss.py``) and a Moore Threads specialization
// (``runtime/backend/_mthreads/fused/moe_load_balance_loss.py``).  Only the
// launch geometry differs between them -- the two files expose the same two
// kernels with the same signatures -- so this wrapper picks the file and the
// geometry together:
//
//   * MUSA builds use the ``_mthreads`` specialization's calibration (a
//     32-token tile, 8 warps) when the source tree carries it;
//   * everything else uses the backend-neutral launch (a 32-token tile, 4
//     warps up to 256 experts, 8 beyond that).
//
// ``select_block_tokens`` / ``select_num_warps`` are direct ports of the Python
// constants; keep the three in sync.

#include <algorithm>
#include <filesystem>
#include <optional>
#include <string>

#include "flag_gems/backend_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

namespace {

  // Launch geometry of the backend-neutral implementation
  // (``flag_gems.fused.moe_load_balance_loss``): one full accelerator vector of
  // tokens per program, and eight warps once the expert tile no longer fits the
  // four-warp budget.
  constexpr int64_t kGenericBlockTokens = 32;
  constexpr int64_t kGenericWideExpertTile = 256;

  // Moore Threads calibration from ``_NUM_WARPS`` / ``_BLOCK_TOKENS``: eight
  // warps over a 32-token tile is best across the measured shapes.
  constexpr int64_t kMusaBlockTokens = 32;
  constexpr int64_t kMusaNumWarps = 8;

  // The Python launchers leave ``num_stages`` unset, i.e. they run with the
  // Triton back-end default (3 on this toolchain).  Both kernels unroll their
  // Top-K loop at compile time, so the value only affects the K/V pipelining
  // depth of the loads; keep the default so the C++ launch matches Python.
  constexpr int kNumStages = 3;

  // True when this build targets the Moore Threads MUSA runtime, i.e. the
  // ``_mthreads`` specialization is the one the Python side would dispatch to.
  bool build_targets_musa() {
#if defined(FLAGGEMS_USE_MUSA)
    return true;
#else
    return false;
#endif
  }

  // Port of ``_validate_inputs`` (checks only; the messages mirror the Python
  // ones so a failing test reports the same reason).
  void validate_inputs(const at::Tensor &gate_logits,
                       int64_t top_k,
                       const std::optional<at::Tensor> &attention_mask) {
    TORCH_CHECK(gate_logits.dim() == 2,
                "gate_logits must have shape [T, N_e], but got ",
                gate_logits.sizes());
    TORCH_CHECK(gate_logits.scalar_type() == at::kHalf || gate_logits.scalar_type() == at::kBFloat16 ||
                    gate_logits.scalar_type() == at::kFloat,
                "gate_logits supports only float16, bfloat16, and float32, got ",
                c10::toString(gate_logits.scalar_type()));

    const int64_t num_tokens = gate_logits.size(0);
    const int64_t num_experts = gate_logits.size(1);
    TORCH_CHECK(num_tokens > 0 && num_experts > 0, "T and N_e must be greater than zero");
    TORCH_CHECK(top_k >= 1 && top_k <= num_experts,
                "top_k must satisfy 1 <= top_k <= N_e, got ",
                top_k,
                " for N_e=",
                num_experts);

    if (attention_mask.has_value()) {
      const at::Tensor &mask = attention_mask.value();
      TORCH_CHECK(mask.numel() == num_tokens,
                  "attention_mask must contain T elements, but got ",
                  mask.numel(),
                  " for T=",
                  num_tokens);
      TORCH_CHECK(mask.scalar_type() == at::kBool || mask.scalar_type() == at::kByte ||
                      mask.scalar_type() == at::kInt || mask.scalar_type() == at::kLong,
                  "attention_mask supports only bool, uint8, int32, and int64, got ",
                  c10::toString(mask.scalar_type()));
      TORCH_CHECK(mask.device() == gate_logits.device(),
                  "gate_logits and attention_mask must be on the same device");
    }
  }

}  // namespace

at::Tensor moe_load_balance_loss(const at::Tensor &gate_logits,
                                 int64_t top_k,
                                 const std::optional<at::Tensor> &attention_mask) {
  validate_inputs(gate_logits, top_k, attention_mask);

  const at::Tensor logits = gate_logits.contiguous();
  const bool has_mask = attention_mask.has_value();
  // The kernels never dereference the mask pointer when HAS_MASK is false, so
  // the logits double as the dummy argument and no extra allocation is needed.
  const at::Tensor flat_mask = has_mask ? attention_mask.value().reshape({-1}).contiguous() : logits;
  const int64_t num_tokens = logits.size(0);
  const int64_t num_experts = logits.size(1);

  // Every atomic target lives in its own allocation: some backends widen vector
  // atomics to a full accelerator vector, which would alias neighbouring
  // buffers otherwise.
  const at::TensorOptions float_options = logits.options().dtype(at::kFloat);
  const at::Tensor expert_counts = at::zeros({num_experts}, float_options);
  const at::Tensor valid_token_count = at::zeros({}, float_options);
  const at::Tensor output = at::zeros({}, float_options);

  const std::filesystem::path src_path = utils::get_flag_gems_src_path();
  const std::filesystem::path musa_kernel_path =
      src_path / "runtime" / "backend" / "_mthreads" / "fused" / "moe_load_balance_loss.py";
  // Vendor-calibrated specialization: only on a MUSA build (where the Python
  // side dispatches to `_mthreads`) and only when the source tree carries it.
  const bool use_musa_kernel = build_targets_musa() && std::filesystem::exists(musa_kernel_path);
  const std::filesystem::path kernel_path =
      use_musa_kernel ? musa_kernel_path : src_path / "fused" / "moe_load_balance_loss.py";

  const int64_t block_experts = utils::next_power_of_2(num_experts);
  const int64_t block_tokens = use_musa_kernel ? kMusaBlockTokens : kGenericBlockTokens;
  const int num_warps =
      static_cast<int>(use_musa_kernel ? kMusaNumWarps : (block_experts <= kGenericWideExpertTile ? 4 : 8));
  const int64_t grid_x = (num_tokens + block_tokens - 1) / block_tokens;

  c10::DeviceGuard guard(logits.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  const TritonJITFunction &count_kernel =
      TritonJITFunction::get_instance(kernel_path.string(), "_topk_count_kernel");
  const TritonJITFunction &loss_kernel =
      TritonJITFunction::get_instance(kernel_path.string(), "_loss_kernel");

  // def _topk_count_kernel(gate_logits, attention_mask, expert_counts,
  //                        valid_token_count, num_tokens, NUM_EXPERTS: tl.constexpr,
  //                        TOP_K: tl.constexpr, BLOCK_TOKENS: tl.constexpr,
  //                        BLOCK_EXPERTS: tl.constexpr, HAS_MASK: tl.constexpr)
  count_kernel(raw_stream,
               grid_x,
               /* grid_y */ 1,
               /* grid_z */ 1,
               num_warps,
               kNumStages,
               logits,
               flat_mask,
               expert_counts,
               valid_token_count,
               num_tokens,
               num_experts,
               top_k,
               block_tokens,
               block_experts,
               has_mask);

  // def _loss_kernel(gate_logits, attention_mask, expert_counts,
  //                  valid_token_count, output, num_tokens,
  //                  NUM_EXPERTS: tl.constexpr, BLOCK_TOKENS: tl.constexpr,
  //                  BLOCK_EXPERTS: tl.constexpr, HAS_MASK: tl.constexpr)
  loss_kernel(raw_stream,
              grid_x,
              /* grid_y */ 1,
              /* grid_z */ 1,
              num_warps,
              kNumStages,
              logits,
              flat_mask,
              expert_counts,
              valid_token_count,
              output,
              num_tokens,
              num_experts,
              block_tokens,
              block_experts,
              has_mask);

  return output;
}

}  // namespace flag_gems
