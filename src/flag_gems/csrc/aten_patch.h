#pragma once
#include <string>
#include <vector>

extern std::vector<std::string> registered_ops;

#define REGISTER_AND_LOG(opname, func) \
  m.impl(opname, TORCH_FN(func));      \
  registered_ops.push_back(opname);
