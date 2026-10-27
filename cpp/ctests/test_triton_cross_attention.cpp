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

// Correctness tests for the C++ wrapped flag_gems::cross_attention.
//
// The reference and the tolerance mirror tests/test_cross_attention.py: the
// result is compared against torch's scaled_dot_product_attention (with
// enable_gqa whenever the head counts differ, and the mask inverted because
// SDPA bool masks mark *allowed* positions while cross_attention blocks
// non-zero entries), using accuracy_utils::gems_assert_close, i.e. the same
// atol/rtol table the Python tests use.

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <ATen/ops/scaled_dot_product_attention.h>

#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "gtest/gtest.h"
#include "torch/torch.h"

namespace {

using flag_gems::accuracy_utils::gems_assert_close;

const std::vector<torch::ScalarType> kDtypes = {torch::kHalf, torch::kFloat, torch::kBFloat16};

torch::TensorOptions tensor_options(const torch::Device &device, torch::ScalarType dtype) {
  return torch::TensorOptions().device(device).dtype(dtype);
}

// Same distribution as the Python test: U(-0.05, 0.05).
torch::Tensor rand_qkv(at::IntArrayRef sizes, const torch::Device &device, torch::ScalarType dtype) {
  auto tensor = torch::empty(sizes, tensor_options(device, dtype));
  tensor.uniform_(-0.05, 0.05);
  return tensor;
}

torch::Tensor sdpa_reference(const torch::Tensor &query,
                             const torch::Tensor &key,
                             const torch::Tensor &value,
                             const std::optional<torch::Tensor> &attn_mask = std::nullopt,
                             const std::optional<double> &scale = std::nullopt) {
  std::optional<torch::Tensor> allowed_mask = std::nullopt;
  if (attn_mask.has_value()) {
    // cross_attention: non-zero == blocked; SDPA: true == allowed.
    allowed_mask = ~(attn_mask.value() != 0);
  }
  return at::scaled_dot_product_attention(query,
                                          key,
                                          value,
                                          allowed_mask,
                                          /*dropout_p=*/0.0,
                                          /*is_causal=*/false,
                                          scale,
                                          /*enable_gqa=*/query.size(1) != key.size(1));
}

std::string describe(int64_t B,
                     int64_t Hq,
                     int64_t Hkv,
                     int64_t Lq,
                     int64_t Lk,
                     int64_t Dqk,
                     int64_t Dv,
                     torch::ScalarType dtype,
                     const std::optional<torch::Tensor> &mask,
                     const std::optional<double> &scale) {
  std::ostringstream oss;
  oss << "B=" << B << " Hq=" << Hq << " Hkv=" << Hkv << " Lq=" << Lq << " Lk=" << Lk << " Dqk=" << Dqk
      << " Dv=" << Dv << " dtype=" << c10::toString(dtype) << " mask=" << (mask.has_value() ? "yes" : "no")
      << " scale=" << (scale.has_value() ? std::to_string(scale.value()) : "default");
  return oss.str();
}

// One full case: build inputs, run the wrapped op, compare with SDPA.
void expect_matches_reference(int64_t B,
                              int64_t Hq,
                              int64_t Hkv,
                              int64_t Lq,
                              int64_t Lk,
                              int64_t Dqk,
                              int64_t Dv,
                              torch::ScalarType dtype,
                              const std::optional<torch::Tensor> &mask = std::nullopt,
                              const std::optional<double> &scale = std::nullopt,
                              const char *tag = "") {
  const torch::Device device = flag_gems::test::default_device();
  auto query = rand_qkv({B, Hq, Lq, Dqk}, device, dtype);
  auto key = rand_qkv({B, Hkv, Lk, Dqk}, device, dtype);
  auto value = rand_qkv({B, Hkv, Lk, Dv}, device, dtype);

  auto out = flag_gems::cross_attention(query, key, value, mask, scale);
  const std::string info = describe(B, Hq, Hkv, Lq, Lk, Dqk, Dv, dtype, mask, scale);
  SCOPED_TRACE(info + (tag[0] ? std::string(" [") + tag + "]" : std::string()));

  EXPECT_TRUE((out.sizes().vec() == std::vector<int64_t> {B, Hq, Lq, Dv})) << info;
  EXPECT_EQ(out.scalar_type(), dtype) << info;

  auto reference = sdpa_reference(query, key, value, mask, scale);
  auto result = gems_assert_close(out, reference, dtype);
  EXPECT_TRUE(result.ok) << info << "\n" << result.message;
}

// `1::3` columns set, like the Python mask tests.
torch::Tensor mask_column_flag(int64_t length, const torch::Device &device, torch::ScalarType mask_dtype) {
  auto cols = at::arange(length, torch::TensorOptions().device(device).dtype(torch::kLong));
  return (cols.remainder(3) == 1).to(mask_dtype);
}

torch::Tensor make_mask(const std::vector<int64_t> &shape,
                        const torch::Device &device,
                        torch::ScalarType mask_dtype) {
  const int64_t last = shape.back();
  auto flag = mask_column_flag(last, device, mask_dtype);
  std::vector<int64_t> view_shape(shape.size(), 1);
  view_shape.back() = last;
  return flag.view(view_shape).expand(shape).contiguous();
}

void expect_throws(const std::function<void()> &fn, const std::string &needle) {
  try {
    fn();
    FAIL() << "expected an error containing '" << needle << "'";
  } catch (const c10::Error &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find(needle), std::string::npos)
        << "expected error mentioning '" << needle << "', got: " << message;
  }
}

}  // namespace

// Dense / GQA / MQA / different value head dim, no mask, default scale.
TEST(TritonCrossAttentionTest, MatchesSdpaReference) {
  // (B, Hq, Hkv, Lq, Lk, Dqk, Dv)
  const std::vector<std::array<int64_t, 7>> shapes = {
      {1, 16, 16,   16,   16, 128, 128}, // single tile, MHA
      {4,  8,  8,  128,  128,  64,  64}, // exact-bounds path
      {2, 16, 16,  256,  128,  64,  64}, // Lq != Lk
      {2, 32, 32,  128,  512, 128, 128}, // head_dim 128
      {1, 16, 16,  257,  513, 128, 128}, // non power-of-two lengths
      {4,  8,  8, 2048,  256,  64,  64}, // larger batch
      {2,  4,  4, 1024, 1034,  64,  64}, // Lk not a multiple of the tile
      {2,  4,  2,  512,  612, 128, 128}, // GQA (Hq / Hkv == 2)
      {2,  4,  1, 1024, 1034,  64,  64}, // MQA (Hq / Hkv == 4)
      {2,  3,  3,   33,   47,  64,  32}, // Dv < Dqk
      {2,  3,  3,   61,  129, 128,  32}, // Dv < Dqk, non power-of-two lengths
  };
  for (auto dtype : kDtypes) {
    for (const auto &s : shapes) {
      expect_matches_reference(s[0], s[1], s[2], s[3], s[4], s[5], s[6], dtype);
    }
  }
}

// Explicit scale, both with and without a mask.
TEST(TritonCrossAttentionTest, CustomScale) {
  const torch::Device device = flag_gems::test::default_device();
  for (auto dtype : kDtypes) {
    expect_matches_reference(2, 3, 3, 5, 7, 32, 32, dtype, std::nullopt, 0.125);
    // scale larger than 1 / sqrt(Dqk)
    expect_matches_reference(2, 3, 3, 5, 7, 32, 32, dtype, std::nullopt, 0.5);
    auto mask = make_mask({2, 1, 5, 7}, device, torch::kBool);
    expect_matches_reference(2, 3, 3, 5, 7, 32, 32, dtype, mask, 0.125);
  }
}

// All four accepted mask layouts, bool and uint8 mask dtypes.
TEST(TritonCrossAttentionTest, MaskContract) {
  const torch::Device device = flag_gems::test::default_device();
  const int64_t B = 2, H = 3, Lq = 5, Lk = 7, D = 32;
  const std::vector<std::vector<int64_t>> mask_shapes = {
      {Lq, Lk},
      { 1, 1,Lq, Lk},
      { B, 1,Lq, Lk},
      { B, H,Lq, Lk}
  };
  for (auto dtype : kDtypes) {
    for (auto mask_dtype : {torch::kBool, torch::kByte}) {
      for (const auto &mask_shape : mask_shapes) {
        auto mask = make_mask(mask_shape, device, mask_dtype);
        expect_matches_reference(B, H, H, Lq, Lk, D, D, dtype, mask, 0.125);
      }
    }
  }
}

// A fully blocked query row must be exactly zero and stay finite.
TEST(TritonCrossAttentionTest, FullyMaskedRowsAreExactZero) {
  const torch::Device device = flag_gems::test::default_device();
  const int64_t B = 2, H = 2, Lq = 5, Lk = 9, D = 32;
  for (auto dtype : kDtypes) {
    auto query = rand_qkv({B, H, Lq, D}, device, dtype);
    auto key = rand_qkv({B, H, Lk, D}, device, dtype);
    auto value = rand_qkv({B, H, Lk, D}, device, dtype);
    auto mask = at::zeros({B, 1, Lq, Lk}, tensor_options(device, torch::kBool));
    mask.index_put_({at::indexing::Slice(), at::indexing::Slice(), 2, at::indexing::Slice()}, true);

    auto out = flag_gems::cross_attention(query, key, value, mask);
    SCOPED_TRACE(std::string("dtype=") + c10::toString(dtype));
    EXPECT_EQ(torch::count_nonzero(out.select(2, 2)).item<int64_t>(), 0);
    EXPECT_TRUE(at::isfinite(out).all().item<bool>());

    auto result = gems_assert_close(out, sdpa_reference(query, key, value, mask), dtype);
    EXPECT_TRUE(result.ok) << result.message;
  }
}

// Non-contiguous query/key/value/mask (sliced last dim, permuted key).
TEST(TritonCrossAttentionTest, NonContiguousQkvAndMask) {
  const torch::Device device = flag_gems::test::default_device();
  const int64_t B = 2, H = 3, Lq = 7, Lk = 11, D = 24;
  for (auto dtype : kDtypes) {
    auto query = rand_qkv({B, H, Lq, D * 2}, device, dtype).slice(3, 0, c10::nullopt, 2);
    auto key = rand_qkv({B, Lk, H, D}, device, dtype).permute({0, 2, 1, 3});
    auto value = rand_qkv({B, H, Lk * 2, D}, device, dtype).slice(2, 0, c10::nullopt, 2);
    auto mask = make_mask({Lq, Lk * 2}, device, torch::kByte).slice(1, 0, c10::nullopt, 2);

    SCOPED_TRACE(std::string("dtype=") + c10::toString(dtype));
    EXPECT_FALSE(query.is_contiguous());
    EXPECT_FALSE(key.is_contiguous());
    EXPECT_FALSE(value.is_contiguous());
    EXPECT_FALSE(mask.is_contiguous());

    auto out = flag_gems::cross_attention(query, key, value, mask, 0.125);
    auto result = gems_assert_close(out, sdpa_reference(query, key, value, mask, 0.125), dtype);
    EXPECT_TRUE(result.ok) << result.message;
  }
}

// The validation contract of the Python op, surfaced as c10::Error.
TEST(TritonCrossAttentionTest, ValidationContract) {
  const torch::Device device = flag_gems::test::default_device();
  auto q = torch::randn({2, 4, 5, 16}, device);
  auto k = torch::randn({2, 4, 7, 16}, device);
  auto v = torch::randn_like(k);

  expect_throws([&] { flag_gems::cross_attention(q[0], k, v); }, "4-dimensional");
  expect_throws([&] { flag_gems::cross_attention(q, k.to(torch::kHalf), v.to(torch::kHalf)); }, "same dtype");
  expect_throws(
      [&] { flag_gems::cross_attention(q.to(torch::kLong), k.to(torch::kLong), v.to(torch::kLong)); },
      "supports only");
  expect_throws([&] { flag_gems::cross_attention(q, k.slice(1, 0, 3), v.slice(1, 0, 3)); },
                "non-zero integer");
  expect_throws([&] { flag_gems::cross_attention(q, k, v.slice(2, 0, -1)); }, "sequence lengths");
  expect_throws([&] { flag_gems::cross_attention(q, k.slice(3, 0, -1), v); }, "query_D == key_D >= value_D");
  expect_throws(
      [&] {
        flag_gems::cross_attention(q, k, v, at::zeros({5, 7}, tensor_options(device, torch::kFloat)));
      },
      "torch.bool or torch.uint8");
  expect_throws(
      [&] {
        flag_gems::cross_attention(q, k, v, at::zeros({1, 4, 5, 7}, tensor_options(device, torch::kBool)));
      },
      "shape must be one of");
  expect_throws(
      [&] { flag_gems::cross_attention(q, k, v, std::nullopt, std::numeric_limits<double>::infinity()); },
      "finite");
  expect_throws(
      [&] {
        auto big = torch::empty({1, 1, 1, 769}, tensor_options(device, torch::kFloat));
        flag_gems::cross_attention(big, big, big);
      },
      "[1, 768]");
}
