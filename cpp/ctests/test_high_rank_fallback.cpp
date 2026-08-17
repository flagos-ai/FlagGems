// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0

#include <flag_gems/operators.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

// Test that rank > POINTWISE_MAX_RANK (5) produces a clear error
// rather than a bare runtime_error
TEST(HighRankOps, AbsRank6NonContiguous) {
  // Create a rank-6 non-contiguous tensor (slicing prevents fast path)
  auto x = torch::randn({2, 3, 4, 5, 6, 8}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
  auto sliced = x.index({"...", torch::indexing::Slice(0, 2)});

  EXPECT_EQ(sliced.dim(), 6);
  EXPECT_FALSE(sliced.is_contiguous());

  // Should throw TORCH_CHECK with message mentioning rank limit
  EXPECT_THROW(
      {
        try {
          flag_gems::abs(sliced);
        } catch (const c10::Error& e) {
          // Verify it's a proper TORCH_CHECK, not bare runtime_error
          std::string msg = e.what();
          EXPECT_NE(msg.find("rank"), std::string::npos);
          EXPECT_NE(msg.find("6"), std::string::npos);
          EXPECT_NE(msg.find("5"), std::string::npos);  // MAX_RANK
          throw;
        }
      },
      c10::Error);
}

TEST(HighRankOps, AbsRank6Contiguous) {
  // Contiguous rank-6 should also fail clearly if it skips fast path
  auto x = torch::randn({2, 2, 2, 2, 2, 2}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
  auto transposed = x.permute({5, 4, 3, 2, 1, 0});  // Force non-fast-path

  EXPECT_EQ(transposed.dim(), 6);

  EXPECT_THROW({ flag_gems::abs(transposed); }, c10::Error);
}
