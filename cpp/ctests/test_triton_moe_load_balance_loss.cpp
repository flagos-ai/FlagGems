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

// Correctness tests for the C++ wrapped flag_gems::moe_load_balance_loss.
//
// The reference and the tolerances mirror tests/test_moe_load_balance_loss.py:
// the loss is computed from the (already rounded) router logits -- softmax for
// the mean routing probabilities and torch.topk for the Top-K assignment
// counts, both in the high-precision reference dtype -- and compared through
// accuracy_utils::gems_assert_close with the same reduce_dim (the expert count)
// the Python test uses.

#include <array>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <ATen/ops/arange.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/topk.h>

#include "flag_gems/accuracy_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/test_utils.h"
#include "gtest/gtest.h"
#include "torch/torch.h"

namespace {

using flag_gems::accuracy_utils::gems_assert_close;
using flag_gems::accuracy_utils::to_reference;

const std::vector<torch::ScalarType> kDtypes = {torch::kHalf, torch::kBFloat16, torch::kFloat};

torch::TensorOptions tensor_options(const torch::Device &device, torch::ScalarType dtype) {
  return torch::TensorOptions().device(device).dtype(dtype);
}

// A column pattern that repeats every `period` tokens starting at `offset`,
// i.e. the ``mask[..., offset::period] = 0`` masks the Python tests build, but
// without in-place index assignment.
torch::Tensor periodic_flag(int64_t num_tokens,
                            int64_t offset,
                            int64_t period,
                            const torch::Device &device,
                            torch::ScalarType dtype) {
  auto index = at::arange(num_tokens, tensor_options(device, torch::kLong));
  auto keep = index.remainder(period).ne(offset % period);
  return keep.to(dtype);
}

// Port of ``_reference`` from the Python test: the statistics are computed in
// the reference dtype chosen by ``to_reference`` (double on CPU for backends
// without fp64, matching the Python helper).
torch::Tensor reference(const torch::Tensor &gate_logits,
                        int64_t top_k,
                        const std::optional<torch::Tensor> &attention_mask = std::nullopt) {
  const torch::Tensor logits = to_reference(gate_logits.detach(), /*upcast=*/true);
  const int64_t num_tokens = logits.size(0);
  const int64_t num_experts = logits.size(1);

  auto probabilities = at::softmax(logits, -1);
  // at::topk returns (values, indices); only the latter feeds the histogram.
  auto selected_experts = std::get<1>(at::topk(logits, top_k, /*dim=*/-1, /*largest=*/true, /*sorted=*/true));

  torch::Tensor valid = torch::ones({num_tokens}, logits.options());
  if (attention_mask.has_value()) {
    valid = to_reference(attention_mask.value().detach().reshape({-1})).to(logits.dtype());
  }

  const double total_valid = valid.sum().item<double>();
  if (total_valid == 0.0) {
    return torch::zeros({}, logits.options());
  }

  auto counts = torch::zeros({num_experts}, logits.options());
  counts.scatter_add_(0, selected_experts.reshape({-1}), valid.repeat_interleave(top_k));
  auto frequency = counts / total_valid;
  auto mean_probability = (probabilities * valid.unsqueeze(-1)).sum(0) / total_valid;
  return static_cast<double>(num_experts) * (frequency * mean_probability).sum();
}

std::string describe(int64_t num_tokens,
                     int64_t num_experts,
                     int64_t top_k,
                     torch::ScalarType dtype,
                     const std::optional<torch::Tensor> &mask) {
  std::ostringstream oss;
  oss << "T=" << num_tokens << " Ne=" << num_experts << " top_k=" << top_k
      << " dtype=" << c10::toString(dtype) << " mask=" << (mask.has_value() ? "yes" : "no");
  return oss.str();
}

void expect_close(const torch::Tensor &out, const torch::Tensor &expected, int64_t num_experts) {
  EXPECT_EQ(out.dim(), 0) << "the loss must be a scalar tensor";
  EXPECT_EQ(out.scalar_type(), torch::kFloat) << "the operator always returns float32";
  // Same tolerance contract as the Python test.
  auto result = gems_assert_close(out, expected, torch::kFloat, /*equal_nan=*/false, num_experts);
  EXPECT_TRUE(result.ok) << result.message;
}

// One full case: build the logits, run the wrapped op, compare with the
// reference.  ``masked`` selects the Python test's ``1::4`` mask layout.
void expect_matches_reference(int64_t num_tokens,
                              int64_t num_experts,
                              int64_t top_k,
                              torch::ScalarType dtype,
                              bool masked,
                              const char *tag = "") {
  const torch::Device device = flag_gems::test::default_device();
  torch::manual_seed(2026);
  auto logits = torch::randn({num_tokens, num_experts}, tensor_options(device, dtype));
  std::optional<torch::Tensor> attention_mask = std::nullopt;
  if (masked) {
    attention_mask = periodic_flag(num_tokens, 1, 4, device, torch::kLong).reshape({1, num_tokens});
  }

  SCOPED_TRACE(describe(num_tokens, num_experts, top_k, dtype, attention_mask) +
               (tag[0] ? std::string(" [") + tag + "]" : std::string()));

  auto out = flag_gems::moe_load_balance_loss(logits, top_k, attention_mask);
  expect_close(out, reference(logits, top_k, attention_mask), num_experts);
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

// The shape/dtype/mask sweep of the Python test.
TEST(TritonMoeLoadBalanceLossTest, MatchesReference) {
  // (num_tokens, num_experts, top_k)
  const std::vector<std::array<int64_t, 3>> shapes = {
      {   1,  1, 1},
      {  17,  8, 2},
      { 257, 16, 4},
      {1024, 64, 2},
  };
  for (auto dtype : kDtypes) {
    for (bool masked : {false, true}) {
      for (const auto &shape : shapes) {
        expect_matches_reference(shape[0], shape[1], shape[2], dtype, masked);
      }
    }
  }
}

// Non-power-of-two token/expert counts with a sparse mask: exercises the
// pipelined counting path without relying on a benchmark-only shape.
TEST(TritonMoeLoadBalanceLossTest, IrregularLargeShape) {
  const torch::Device device = flag_gems::test::default_device();
  const int64_t num_tokens = 2051, num_experts = 96, top_k = 5;
  for (auto dtype : kDtypes) {
    torch::manual_seed(2026);
    auto logits = torch::randn({num_tokens, num_experts}, tensor_options(device, dtype));
    auto attention_mask = periodic_flag(num_tokens, 5, 7, device, torch::kBool);

    SCOPED_TRACE(std::string("dtype=") + c10::toString(dtype));
    auto out = flag_gems::moe_load_balance_loss(logits, top_k, attention_mask);
    expect_close(out, reference(logits, top_k, attention_mask), num_experts);
  }
}

// Non-contiguous logits (strided expert dim) and a non-contiguous uint8 mask.
TEST(TritonMoeLoadBalanceLossTest, NonContiguousLogitsAndMask) {
  const torch::Device device = flag_gems::test::default_device();
  const int64_t num_tokens = 97, num_experts = 24, top_k = 3;
  for (auto dtype : kDtypes) {
    torch::manual_seed(2027);
    auto logits =
        torch::randn({num_tokens, num_experts}, tensor_options(device, dtype)).slice(1, 0, c10::nullopt, 2);
    const auto flag = periodic_flag(num_tokens, 0, 5, device, torch::kByte);
    auto mask_storage = at::stack({flag, flag}, 1);
    auto attention_mask = mask_storage.select(1, 0);

    SCOPED_TRACE(std::string("dtype=") + c10::toString(dtype));
    EXPECT_FALSE(logits.is_contiguous());
    EXPECT_FALSE(attention_mask.is_contiguous());

    auto out = flag_gems::moe_load_balance_loss(logits, top_k, attention_mask);
    expect_close(out, reference(logits, top_k, attention_mask), num_experts);
  }
}

// An all-zero mask makes the operator return exactly zero, not just a small
// value: the denominator is clamped to one and every token drops out.
TEST(TritonMoeLoadBalanceLossTest, AllMaskedTokensReturnZero) {
  const torch::Device device = flag_gems::test::default_device();
  for (auto dtype : kDtypes) {
    auto logits = torch::randn({33, 8}, tensor_options(device, dtype));
    auto attention_mask = torch::zeros({33}, tensor_options(device, torch::kBool));

    SCOPED_TRACE(std::string("dtype=") + c10::toString(dtype));
    auto out = flag_gems::moe_load_balance_loss(logits, 2, attention_mask);
    EXPECT_EQ(out.dim(), 0);
    EXPECT_EQ(out.item<float>(), 0.0f);
  }
}

// top_k defaults to two, and the mask argument may be omitted entirely.
TEST(TritonMoeLoadBalanceLossTest, DefaultTopKIsTwo) {
  const torch::Device device = flag_gems::test::default_device();
  torch::manual_seed(2026);
  auto logits = torch::randn({19, 8}, tensor_options(device, torch::kFloat));

  auto out = flag_gems::moe_load_balance_loss(logits);
  expect_close(out, reference(logits, 2), logits.size(1));
}

// The validation contract of the Python op, surfaced as c10::Error.
TEST(TritonMoeLoadBalanceLossTest, ValidationContract) {
  const torch::Device device = flag_gems::test::default_device();
  auto logits = torch::randn({8, 4}, tensor_options(device, torch::kFloat));

  expect_throws([&] { flag_gems::moe_load_balance_loss(logits.unsqueeze(0)); }, "shape [T, N_e]");
  expect_throws([&] { flag_gems::moe_load_balance_loss(logits.to(torch::kLong)); }, "supports only");
  expect_throws(
      [&] {
        flag_gems::moe_load_balance_loss(torch::empty({0, 4}, tensor_options(device, torch::kFloat)));
      },
      "greater than zero");
  expect_throws([&] { flag_gems::moe_load_balance_loss(logits, 0); }, "1 <= top_k <= N_e");
  expect_throws([&] { flag_gems::moe_load_balance_loss(logits, 5); }, "1 <= top_k <= N_e");
  expect_throws(
      [&] {
        flag_gems::moe_load_balance_loss(logits, 2, torch::ones({7}, tensor_options(device, torch::kBool)));
      },
      "must contain T elements");
  expect_throws(
      [&] {
        flag_gems::moe_load_balance_loss(logits, 2, torch::ones({8}, tensor_options(device, torch::kFloat)));
      },
      "attention_mask supports only");
  expect_throws(
      [&] {
        flag_gems::moe_load_balance_loss(logits,
                                         2,
                                         torch::ones({8}, torch::TensorOptions().dtype(torch::kBool)));
      },
      "same device");
}
