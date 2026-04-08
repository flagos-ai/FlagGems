#include "aten_patch.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "flag_gems/operators.h"
#include "torch/python.h"

std::vector<std::string> registered_ops;

// 用于动态注册的 Library 对象
static std::unique_ptr<torch::Library> aten_lib;

// 确保 Library 对象已初始化
static torch::Library& get_aten_lib() {
  if (!aten_lib) {
    aten_lib = std::make_unique<torch::Library>(torch::Library::IMPL,
                                                std::string("aten"),
                                                c10::make_optional(c10::DispatchKey::CUDA),
                                                __FILE__,
                                                __LINE__);
  }
  return *aten_lib;
}

// 定义算子注册表：算子名 -> 注册函数
using RegisterFunc = std::function<void(torch::Library&)>;
static std::unordered_map<std::string, RegisterFunc>& get_op_registry() {
  static std::unordered_map<std::string, RegisterFunc> registry = {
      { "max.dim_max",[](torch::Library& m) { m.impl("max.dim_max", TORCH_FN(flag_gems::max_dim_max)); }                      },
      {     "max.dim",           [](torch::Library& m) { m.impl("max.dim", TORCH_FN(flag_gems::max_dim)); }},
      {         "max",                   [](torch::Library& m) { m.impl("max", TORCH_FN(flag_gems::max)); }},
      {         "sum",                   [](torch::Library& m) { m.impl("sum", TORCH_FN(flag_gems::sum)); }},
      {       "zeros",               [](torch::Library& m) { m.impl("zeros", TORCH_FN(flag_gems::zeros)); }},
#ifdef FLAGGEMS_POINTWISE_DYNAMIC
      {  "add.Tensor",     [](torch::Library& m) { m.impl("add.Tensor", TORCH_FN(flag_gems::add_tensor)); }},
      { "add_.Tensor",
       [](torch::Library& m) { m.impl("add_.Tensor", TORCH_FN(flag_gems::add_tensor_inplace)); }           },
      {  "add.Scalar",     [](torch::Library& m) { m.impl("add.Scalar", TORCH_FN(flag_gems::add_scalar)); }},
      { "add_.Scalar",
       [](torch::Library& m) { m.impl("add_.Scalar", TORCH_FN(flag_gems::add_scalar_inplace)); }           },
      { "fill.Scalar",   [](torch::Library& m) { m.impl("fill.Scalar", TORCH_FN(flag_gems::fill_scalar)); }},
      {"fill_.Scalar", [](torch::Library& m) { m.impl("fill_.Scalar", TORCH_FN(flag_gems::fill_scalar_)); }},
      { "fill.Tensor",   [](torch::Library& m) { m.impl("fill.Tensor", TORCH_FN(flag_gems::fill_tensor)); }},
      {"fill_.Tensor", [](torch::Library& m) { m.impl("fill_.Tensor", TORCH_FN(flag_gems::fill_tensor_)); }},
#endif
  };
  return registry;
}

std::vector<std::string> get_registered_ops() {
  return registered_ops;
}

std::vector<std::string> get_available_ops() {
  std::vector<std::string> ops;
  for (const auto& pair : get_op_registry()) {
    ops.push_back(pair.first);
  }
  return ops;
}

// 动态注册指定的算子
void register_cpp_ops(const std::vector<std::string>& op_names) {
  auto& lib = get_aten_lib();
  auto& registry = get_op_registry();
  std::unordered_set<std::string> already_registered(registered_ops.begin(), registered_ops.end());

  for (const auto& name : op_names) {
    if (already_registered.count(name)) continue;  // 避免重复注册

    auto it = registry.find(name);
    if (it != registry.end()) {
      it->second(lib);
      registered_ops.push_back(name);
    }
  }
}

// 注册所有可用算子
void register_all_cpp_ops() {
  register_cpp_ops(get_available_ops());
}

PYBIND11_MODULE(aten_patch, m) {
  m.def("get_registered_ops", &get_registered_ops, "Get list of registered cpp wrapper ops");
  m.def("get_available_ops", &get_available_ops, "Get list of all available cpp wrapper ops");
  m.def("register_cpp_ops", &register_cpp_ops, "Register specified cpp wrapper ops");
  m.def("register_all_cpp_ops", &register_all_cpp_ops, "Register all available cpp wrapper ops");
}
