// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0

#include <flag_gems/operators.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "flag_gems/accuracy_utils.h"

// Test remainder with SCALAR first, tensor second (Scalar_Tensor overload)
// This exercises the rem_st kernel with is_tensor=[False, True]
TEST(RemainderScalarFirst, Basic) {
    auto b = torch::randn({10}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    double a_scalar = 3.5;

    auto result = flag_gems::remainder_st(a_scalar, b);

    // Reference: compute on CPU
    auto b_cpu = b.cpu();
    auto ref = torch::remainder(a_scalar, b_cpu);

    auto check = flag_gems::accuracy_utils::gems_assert_close(result, ref);
    EXPECT_TRUE(check.ok) << check.message;
}

TEST(RemainderScalarFirst, ZeroDimTensor) {
    // Both operands 0-dim: should take host path in boxed adapter
    auto b = torch::tensor(2.0, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    double a_scalar = 7.5;

    auto result = flag_gems::remainder_st(a_scalar, b);

    auto b_cpu = b.cpu();
    auto ref = torch::remainder(a_scalar, b_cpu);

    auto check = flag_gems::accuracy_utils::gems_assert_close(result, ref);
    EXPECT_TRUE(check.ok) << check.message;
}
