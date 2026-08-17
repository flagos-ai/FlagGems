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

#include <gtest/gtest.h>
#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "torch/torch.h"

// ==============================================================================
// Tests for the unary elementwise ops routed through pointwise_dynamic:
//   abs, neg, exp, sqrt, rsqrt, tanh, sigmoid, silu, relu, gelu
//
// Each op is checked against its ATen reference across several ranks,
// non-contiguous inputs, empty tensors, and half/bfloat16 dtypes.
// ==============================================================================

namespace {

using flag_gems::accuracy_utils::gems_assert_close;

class UnaryTest : public ::testing::Test {
 protected:
  const torch::Device device = flag_gems::test::default_device();

  // Positive inputs for ops whose real domain excludes negatives (sqrt/rsqrt).
  torch::Tensor pos(std::vector<int64_t> shape, torch::Dtype dtype = torch::kFloat) {
    return torch::rand(shape, torch::TensorOptions(device).dtype(dtype)) + 0.5;
  }
  torch::Tensor any(std::vector<int64_t> shape, torch::Dtype dtype = torch::kFloat) {
    return torch::randn(shape, torch::TensorOptions(device).dtype(dtype));
  }
};

// Helper: assert out matches ref in both shape and value.
void expect_close(const at::Tensor& out, const at::Tensor& ref) {
  ASSERT_EQ(out.sizes(), ref.sizes());
  ASSERT_EQ(out.scalar_type(), ref.scalar_type());
  auto result = gems_assert_close(out, ref);
  EXPECT_TRUE(result.ok) << result.message;
}

}  // namespace

// ------------------------------------------------------------------ abs
TEST_F(UnaryTest, Abs) {
  auto x2 = any({10, 10});
  auto x2_ref = flag_gems::accuracy_utils::to_reference(x2);
  expect_close(flag_gems::abs(x2), x2_ref.abs());
  auto x = any({4, 5, 6});
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::abs(x), x_ref.abs());
}

// ------------------------------------------------------------------ neg
TEST_F(UnaryTest, Neg) {
  auto x = any({4, 5, 6});
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::neg(x), x_ref.neg());
}

// ------------------------------------------------------------------ exp
TEST_F(UnaryTest, Exp) {
  auto x = any({1024});
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::exp(x), x_ref.exp());
}

// ------------------------------------------------------------------ sqrt
TEST_F(UnaryTest, Sqrt) {
  auto x = pos({8, 16});
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::sqrt(x), x_ref.sqrt());
}

// ------------------------------------------------------------------ rsqrt
TEST_F(UnaryTest, Rsqrt) {
  auto x = pos({8, 16});
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::rsqrt(x), x_ref.rsqrt());
}

// ------------------------------------------------------------------ tanh
TEST_F(UnaryTest, Tanh) {
  auto x = any({2, 3, 4, 5});
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::tanh(x), x_ref.tanh());
}

// ------------------------------------------------------------------ sigmoid
TEST_F(UnaryTest, Sigmoid) {
  auto x = any({2, 3, 4, 5});
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::sigmoid(x), x_ref.sigmoid());
}

// ------------------------------------------------------------------ silu
TEST_F(UnaryTest, Silu) {
  auto x = any({128, 64});
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::silu(x), torch::silu(x_ref));
}

// ------------------------------------------------------------------ relu
TEST_F(UnaryTest, Relu) {
  auto x = any({128, 64});
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::relu(x), torch::relu(x_ref));
}

// ------------------------------------------------------------------ gelu
TEST_F(UnaryTest, GeluNone) {
  auto x = any({128, 64});
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::gelu(x), torch::gelu(x_ref, "none"));
}

TEST_F(UnaryTest, GeluTanh) {
  auto x = any({128, 64});
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::gelu(x, "tanh"), torch::gelu(x_ref, "tanh"));
}

// ------------------------------------------------------------------ non-contiguous
TEST_F(UnaryTest, NonContiguous) {
  auto x = any({4, 5}).t();  // 5x4, non-contiguous
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::neg(x), x_ref.neg());
  expect_close(flag_gems::abs(x), x_ref.abs());
}

// ------------------------------------------------------------------ empty
TEST_F(UnaryTest, EmptyTensor) {
  auto x = any({0, 4});
  auto out = flag_gems::neg(x);
  EXPECT_EQ(out.numel(), 0);
  EXPECT_EQ(out.sizes(), x.sizes());
}

// ------------------------------------------------------------------ half / bfloat16
TEST_F(UnaryTest, Float16) {
  auto x = any({10, 10}, torch::kHalf);
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::silu(x), torch::silu(x_ref));
}

TEST_F(UnaryTest, BFloat16) {
  auto x = any({10, 10}, torch::kBFloat16);
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  expect_close(flag_gems::gelu(x), torch::gelu(x_ref, "none"));
}

// ------------------------------------------------------------------ int -> float promotion
TEST_F(UnaryTest, ExpIntToFloat) {
  auto x = torch::randint(0, 5, {10}, torch::TensorOptions(device).dtype(torch::kInt));
  auto out = flag_gems::exp(x);
  auto x_ref = flag_gems::accuracy_utils::to_reference(x);
  auto ref = x_ref.exp();
  EXPECT_EQ(out.scalar_type(), ref.scalar_type());
  expect_close(out, ref);
}
