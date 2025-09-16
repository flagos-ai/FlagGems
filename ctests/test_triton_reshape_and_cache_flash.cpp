#include <gtest/gtest.h>
#include <torch/torch.h>
#include <tuple>
#include "flag_gems/operators.h"

void reference_reshape_and_cache_flash(const torch::Tensor& key,
                                       const torch::Tensor& value,
                                       torch::Tensor& key_cache,
                                       torch::Tensor& value_cache,
                                       const torch::Tensor& slot_mapping) {
  auto block_size = key_cache.size(1);
  auto num_tokens = key.size(0);

  auto slot_mapping_cpu = slot_mapping.to(torch::kCPU);
  auto block_indices = torch::div(slot_mapping_cpu, block_size, "floor");
  auto block_offsets = slot_mapping_cpu % block_size;

  auto block_indices_acc = block_indices.accessor<int64_t, 1>();
  auto block_offsets_acc = block_offsets.accessor<int64_t, 1>();

  for (int64_t i = 0; i < num_tokens; ++i) {
    auto block_idx = block_indices_acc[i];
    auto block_offset = block_offsets_acc[i];
    key_cache.index_put_({block_idx, block_offset}, key[i]);
    value_cache.index_put_({block_idx, block_offset}, value[i]);
  }
}

class ReshapeAndCacheFlashTest
    : public ::testing::TestWithParam<std::tuple<int64_t, int64_t, int64_t, int64_t, torch::ScalarType>> {};

TEST_P(ReshapeAndCacheFlashTest, CompareWithPureTorchReference) {
  auto [num_tokens, num_heads, head_size, block_size, dtype] = GetParam();

  ASSERT_TRUE(torch::cuda::is_available());
  torch::Device device(torch::kCUDA);
  auto options = torch::TensorOptions().device(device).dtype(dtype);
  auto long_options = options.dtype(torch::kLong);

  const int64_t num_blocks = (num_tokens + block_size - 1) / block_size + 10;

  auto key = torch::randn({num_tokens, num_heads, head_size}, options);
  auto value = torch::randn({num_tokens, num_heads, head_size}, options);

  auto slots = torch::randperm(num_blocks * block_size, long_options);
  auto slot_mapping = slots.slice(0, 0, num_tokens);

  // auto k_scale = torch::tensor({1.0}, options);
  // auto v_scale = torch::tensor({1.0}, options);
  auto k_scale = (key.amax() / 64.0).to(torch::kFloat32);
  auto v_scale = (value.amax() / 64.0).to(torch::kFloat32);

  auto key_cache_initial = torch::randn({num_blocks, block_size, num_heads, head_size}, options);
  auto value_cache_initial = torch::randn({num_blocks, block_size, num_heads, head_size}, options);

  auto key_cache_ref = key_cache_initial.clone();
  auto value_cache_ref = value_cache_initial.clone();
  auto key_cache_test = key_cache_initial.clone();
  auto value_cache_test = value_cache_initial.clone();

  reference_reshape_and_cache_flash(key, value, key_cache_ref, value_cache_ref, slot_mapping);

  std::string kv_cache_dtype("auto");

  flag_gems::reshape_and_cache_flash(key,
                                     value,
                                     key_cache_test,
                                     value_cache_test,
                                     slot_mapping,
                                     kv_cache_dtype,
                                     k_scale,
                                     v_scale);

  double atol = (dtype == torch::kFloat16 || dtype == torch::kBFloat16) ? 1e-2 : 1e-5;
  double rtol = (dtype == torch::kFloat16 || dtype == torch::kBFloat16) ? 1e-2 : 1e-3;

  ASSERT_TRUE(torch::allclose(key_cache_ref, key_cache_test, rtol, atol));
  ASSERT_TRUE(torch::allclose(value_cache_ref, value_cache_test, rtol, atol));
}

INSTANTIATE_TEST_SUITE_P(ReshapeAndCacheFlashTests,
                         ReshapeAndCacheFlashTest,
                         ::testing::Combine(
                             // num_tokens
                             ::testing::Values(42),
                             // num_heads
                             ::testing::Values(8),
                             // head_size
                             ::testing::Values(64, 120, 256),
                             // block_size
                             ::testing::Values(16, 32),
                             // dtype
                             ::testing::Values(torch::kFloat16, torch::kBFloat16, torch::kFloat32)));
