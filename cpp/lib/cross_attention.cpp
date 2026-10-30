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

// C++ wrapper for ``flag_gems.cross_attention`` (BNSD forward-only dense cross
// attention, MHA / GQA / MQA).
//
// The Python side is split into a backend-neutral kernel
// (``ops/cross_attention.py``) and a T-Head ZW810 specialization
// (``runtime/backend/_thead/ops/cross_attention.py``, the configuration
// calibrated for that hardware: bf16-split fp32 MMAs, tiles and pipeline
// depths from a shared-memory budget rule, and the ``EXACT_BOUNDS`` predicate
// elision).  The wrapper selects the specialization only when the running
// device *is* a T-Head PPU -- the ZW810 reports itself as ``PPU-...``, the
// device name flag_gems maps to its ``thead`` vendor -- and the specialization
// is present in the source tree.  Everything else (any other vendor, non-CUDA
// builds) runs the backend-neutral kernel with a conservative tile, so
// vendor-calibrated launch parameters are never used on hardware they were not
// tuned for.
//
// ``select_ppu_tiling`` / ``exact_bounds`` are direct ports of
// ``_select_ppu_tiling`` / ``_exact_bounds`` from the Python specialization;
// keep the two in sync.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>

#if defined(FLAGGEMS_USE_CUDA)
#include <ATen/cuda/CUDAContext.h>
#endif

#include "flag_gems/backend_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

namespace {

  // Bounds and defaults mirrored from flag_gems.ops.cross_attention.
  constexpr int64_t kMaxBatch = 2 * 1024 * 1024;
  constexpr int64_t kMaxHeads = 256;
  constexpr int64_t kMaxSequence = 1024 * 1024;
  constexpr int64_t kMaxHeadDim = 768;
  // Multi-processors on the ZW810; decides whether a launch is too small to fill
  // the device (and therefore wants narrower tiles).
  constexpr int64_t kPpuSmCount = 64;

  // True when `device` is a T-Head ZW810 PPU.  The PPU is driven through the
  // CUDA-compatible API and reports itself as "PPU-..." (e.g. "PPU-ZW810E"),
  // which is the device name flag_gems' vendor detection maps to `thead`.
  // Builds that are not CUDA (NPU, MUSA, ...) have no PPU and therefore always
  // take the backend-neutral path.
  bool is_thead_ppu_device(const at::Device &device) {
#if defined(FLAGGEMS_USE_CUDA)
    const c10::DeviceIndex index = device.has_index() ? device.index() : at::cuda::current_device();
    const cudaDeviceProp *props = at::cuda::getDeviceProperties(index);
    return props != nullptr && std::strncmp(props->name, "PPU", 3) == 0;
#else
    (void)device;
    return false;
#endif
  }

  struct Tiling {
    int64_t block_m;
    int64_t block_n;
    int64_t num_warps;
    int64_t pipeline_ns;
  };

  inline int64_t cdiv(int64_t a, int64_t b) {
    return (a + b - 1) / b;
  }

  // Broadcast strides of the accepted attention-mask layouts, i.e.
  // ``(Lq, Lk)``, ``(1, 1, Lq, Lk)``, ``(B, 1, Lq, Lk)`` and
  // ``(B, Hq, Lq, Lk)``.  A dimension of extent 1 gets stride 0 so the kernel
  // reads the same element for every index it broadcasts over.  Port of
  // ``_mask_strides``.
  std::array<int64_t, 4> mask_strides_of(const at::Tensor &mask) {
    if (mask.dim() == 2) {
      return {0, 0, mask.stride(0), mask.stride(1)};
    }
    std::array<int64_t, 4> strides = {mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3)};
    if (mask.size(0) == 1) {
      strides[0] = 0;
    }
    if (mask.size(1) == 1) {
      strides[1] = 0;
    }
    return strides;
  }

  // Port of ``_select_ppu_tiling`` (ZW810 calibration).
  Tiling select_ppu_tiling(int64_t query_len,
                           int64_t key_len,
                           int64_t qk_dim,
                           int64_t value_dim,
                           bool has_mask,
                           bool is_fp32,
                           int64_t batch_size,
                           int64_t query_heads) {
    const int64_t head_dim = std::max(qk_dim, value_dim);
    const int64_t ns_flat = 1;

    if (is_fp32) {
      // fp32 runs as three bf16 MMAs over a hi/lo split, so a narrow BLOCK_N
      // keeps both MMAs fed while halving the score/P tiles and the softmax ALU
      // work per iteration.
      if (query_len <= 32) {
        return {32, 64, 4, 2};
      }
      if (head_dim >= 96) {
        return {32, 64, 4, 1};
      }
      const int64_t grid_ctas = cdiv(query_len, 64) * batch_size * query_heads;
      if (has_mask && grid_ctas < 4 * kPpuSmCount) {
        return {32, 64, 4, 1};
      }
      return {64, 32, 4, 2};
    }

    if (query_len <= 32) {
      return {32, 64, 4, 3};
    }
    if (head_dim > 128) {
      return {head_dim > 256 ? 16 : 32, 32, 4, ns_flat};
    }
    if (head_dim >= 96) {
      // D = 128: 32 KiB per K/V stage, so at most two pipeline stages fit and a
      // single stage wins while the KV loop is short.
      if (has_mask) {
        return {64, 64, 4, 1};
      }
      const int64_t block_m = query_len >= 4096 ? 128 : 64;
      return {block_m, 64, 4, key_len <= 1024 ? 1 : 2};
    }
    const int64_t ns = 3;
    if (has_mask) {
      const int64_t grid_ctas = cdiv(query_len, 64) * batch_size * query_heads;
      if (grid_ctas < 4 * kPpuSmCount) {
        return {64, 64, 4, 4};
      }
      return {64, 32, 4, 4};
    }
    if (head_dim == 64) {
      return {64, 32, 4, ns};
    }
    return {64, 64, 4, ns};
  }

  // Fallback used with the backend-neutral kernel: correct for every shape
  // (that kernel masks all lanes), only slower than the calibrated table.
  Tiling select_generic_tiling() {
    return {64, 64, 4, 2};
  }

  // Port of ``_exact_bounds``: true when every tile of the launch is fully
  // inside the tensors, which lets the kernel drop all per-lane predicates.
  bool exact_bounds(int64_t query_len,
                    int64_t key_len,
                    int64_t qk_dim,
                    int64_t value_dim,
                    const Tiling &tiling,
                    int64_t qk_block,
                    int64_t value_block) {
    return query_len % tiling.block_m == 0 && key_len % tiling.block_n == 0 && qk_dim == qk_block &&
           value_dim % value_block == 0;
  }

  // Port of ``_validate_inputs`` (checks only; the messages mirror the Python
  // ones so a failing test reports the same reason).
  void validate_inputs(const at::Tensor &query,
                       const at::Tensor &key,
                       const at::Tensor &value,
                       const std::optional<at::Tensor> &attn_mask,
                       const std::optional<double> &scale) {
    const std::pair<std::string, const at::Tensor *> qkv[] = {
        {"query", &query},
        {  "key",   &key},
        {"value", &value}
    };
    for (const auto &[name, tensor] : qkv) {
      TORCH_CHECK(tensor->dim() == 4,
                  name,
                  " must be 4-dimensional in BNSD layout, got ",
                  tensor->dim(),
                  " dimensions");
    }
    TORCH_CHECK(query.device() == key.device() && query.device() == value.device(),
                "query, key, and value must be on the same device");
    TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(),
                "query, key, and value must have the same dtype");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16 ||
                    query.scalar_type() == at::kFloat,
                "cross_attention supports only torch.float16, torch.bfloat16, and torch.float32, got ",
                c10::toString(query.scalar_type()));

    const int64_t batch = query.size(0);
    const int64_t query_heads = query.size(1);
    const int64_t query_len = query.size(2);
    const int64_t qk_dim = query.size(3);
    const int64_t kv_heads = key.size(1);
    const int64_t key_len = key.size(2);
    const int64_t value_dim = value.size(3);

    TORCH_CHECK(batch >= 1 && batch <= kMaxBatch, "batch size must be in [1, ", kMaxBatch, "], got ", batch);
    TORCH_CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0),
                "query, key, and value batch sizes must match");
    TORCH_CHECK(query_heads >= 1 && query_heads <= kMaxHeads,
                "query head count must be in [1, ",
                kMaxHeads,
                "], got ",
                query_heads);
    TORCH_CHECK(key.size(1) == value.size(1) && kv_heads > 0 && query_heads % kv_heads == 0,
                "key/value head counts must match and query_heads / kv_heads must be a non-zero integer");
    TORCH_CHECK(query_len >= 1 && query_len <= kMaxSequence,
                "query sequence length must be in [1, ",
                kMaxSequence,
                "], got ",
                query_len);
    TORCH_CHECK(key_len >= 1 && key_len <= kMaxSequence,
                "key sequence length must be in [1, ",
                kMaxSequence,
                "], got ",
                key_len);
    TORCH_CHECK(key_len == value.size(2), "key and value sequence lengths must match");
    TORCH_CHECK(query.size(3) == key.size(3) && key.size(3) >= value_dim,
                "head dimensions must satisfy query_D == key_D >= value_D");
    TORCH_CHECK(qk_dim >= 1 && qk_dim <= kMaxHeadDim && value_dim >= 1 && value_dim <= kMaxHeadDim,
                "query/key and value head dimensions must be in [1, ",
                kMaxHeadDim,
                "]");

    if (attn_mask.has_value()) {
      const at::Tensor &mask = attn_mask.value();
      TORCH_CHECK(mask.device() == query.device(), "attn_mask must be on the same device as query");
      TORCH_CHECK(mask.scalar_type() == at::kBool || mask.scalar_type() == at::kByte,
                  "attn_mask dtype must be torch.bool or torch.uint8");
      const std::vector<std::vector<int64_t>> allowed = {
          {query_len,     key_len},
          { 1, 1,    query_len, key_len},
          { batch, 1,    query_len, key_len},
          { batch, query_heads,    query_len, key_len}
      };
      bool shape_ok = false;
      for (const auto &shape : allowed) {
        if (mask.sizes().vec() == shape) {
          shape_ok = true;
          break;
        }
      }
      TORCH_CHECK(shape_ok,
                  "attn_mask shape must be one of (Lq, Lk), (1, 1, Lq, Lk), (B, 1, Lq, Lk), "
                  "(B, Hq, Lq, Lk), got ",
                  mask.sizes());
    }
    if (scale.has_value()) {
      TORCH_CHECK(std::isfinite(scale.value()), "scale must be finite, got ", scale.value());
    }
  }

}  // namespace

at::Tensor cross_attention(const at::Tensor &query,
                           const at::Tensor &key,
                           const at::Tensor &value,
                           const std::optional<at::Tensor> &attn_mask,
                           const std::optional<double> &scale) {
  validate_inputs(query, key, value, attn_mask, scale);

  const int64_t batch = query.size(0);
  const int64_t query_heads = query.size(1);
  const int64_t query_len = query.size(2);
  const int64_t qk_dim = query.size(3);
  const int64_t kv_heads = key.size(1);
  const int64_t key_len = key.size(2);
  const int64_t value_dim = value.size(3);
  const double softmax_scale =
      scale.has_value() ? scale.value() : 1.0 / std::sqrt(static_cast<double>(qk_dim));

  at::Tensor output = at::empty({batch, query_heads, query_len, value_dim}, query.options());

  // Without a mask the kernel still takes a (dummy) mask pointer; it is never
  // dereferenced because HAS_MASK is false.
  const bool has_mask = attn_mask.has_value();
  at::Tensor mask_arg = has_mask ? attn_mask.value() : query;
  const std::array<int64_t, 4> mask_strides =
      has_mask ? mask_strides_of(mask_arg) : std::array<int64_t, 4> {0, 0, 0, 0};

  const int64_t qk_block = std::max<int64_t>(16, utils::next_power_of_2(qk_dim));
  const int64_t value_block =
      std::max<int64_t>(16, utils::next_power_of_2(std::min<int64_t>(value_dim, 128)));

  const std::filesystem::path src_path = utils::get_flag_gems_src_path();
  const std::filesystem::path ppu_kernel_path =
      src_path / "runtime" / "backend" / "_thead" / "ops" / "cross_attention.py";
  // Vendor-calibrated specialization: only on the T-Head PPU and only when the
  // source tree actually carries it (see is_thead_ppu_device).
  const bool use_ppu_kernel = is_thead_ppu_device(query.device()) && std::filesystem::exists(ppu_kernel_path);

  const Tiling tiling = use_ppu_kernel ? select_ppu_tiling(query_len,
                                                           key_len,
                                                           qk_dim,
                                                           value_dim,
                                                           has_mask,
                                                           query.scalar_type() == at::kFloat,
                                                           batch,
                                                           query_heads)
                                       : select_generic_tiling();
  const bool use_exact_bounds =
      use_ppu_kernel && exact_bounds(query_len, key_len, qk_dim, value_dim, tiling, qk_block, value_block);
  // fp32 inputs are emulated with three bf16 MMAs over a hi/lo operand split.
  const bool fp32_split = use_ppu_kernel && query.scalar_type() == at::kFloat;

  const int64_t grid_x = cdiv(query_len, tiling.block_m);
  const int64_t grid_y = batch * query_heads;
  const int64_t grid_z = cdiv(value_dim, value_block);
  const int num_warps = static_cast<int>(tiling.num_warps);
  const int num_stages = static_cast<int>(std::max<int64_t>(1, tiling.pipeline_ns));

  c10::DeviceGuard guard(query.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  if (use_ppu_kernel) {
    const TritonJITFunction &kernel =
        TritonJITFunction::get_instance(ppu_kernel_path.string(), "cross_attention_ppu_kernel");
    kernel(raw_stream,
           grid_x,
           grid_y,
           grid_z,
           num_warps,
           num_stages,
           query,
           key,
           value,
           mask_arg,
           output,
           static_cast<float>(softmax_scale),
           query.stride(0),
           query.stride(1),
           query.stride(2),
           query.stride(3),
           key.stride(0),
           key.stride(1),
           key.stride(2),
           key.stride(3),
           value.stride(0),
           value.stride(1),
           value.stride(2),
           value.stride(3),
           mask_strides[0],
           mask_strides[1],
           mask_strides[2],
           mask_strides[3],
           output.stride(0),
           output.stride(1),
           output.stride(2),
           output.stride(3),
           query_heads,
           kv_heads,
           query_len,
           key_len,
           qk_dim,
           value_dim,
           qk_block,
           value_block,
           has_mask,
           tiling.block_m,
           tiling.block_n,
           tiling.pipeline_ns,
           fp32_split,
           use_exact_bounds);
  } else {
    const TritonJITFunction &kernel =
        TritonJITFunction::get_instance((src_path / "ops" / "cross_attention.py").string(),
                                        "cross_attention_fwd_kernel");
    kernel(raw_stream,
           grid_x,
           grid_y,
           grid_z,
           num_warps,
           num_stages,
           query,
           key,
           value,
           mask_arg,
           output,
           static_cast<float>(softmax_scale),
           query.stride(0),
           query.stride(1),
           query.stride(2),
           query.stride(3),
           key.stride(0),
           key.stride(1),
           key.stride(2),
           key.stride(3),
           value.stride(0),
           value.stride(1),
           value.stride(2),
           value.stride(3),
           mask_strides[0],
           mask_strides[1],
           mask_strides[2],
           mask_strides[3],
           output.stride(0),
           output.stride(1),
           output.stride(2),
           output.stride(3),
           query_heads,
           kv_heads,
           query_len,
           key_len,
           qk_dim,
           value_dim,
           qk_block,
           value_block,
           has_mask,
           tiling.block_m,
           tiling.block_n,
           tiling.pipeline_ns);
  }
  return output;
}

}  // namespace flag_gems
