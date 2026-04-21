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
// Define dispatch key based on backend
// CUDA and IX use CUDA dispatch key (IX is CUDA-compatible)
// NPU and MUSA use PrivateUse1 dispatch key
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
    auto dispath_key = c10::DispatchKey::CUDA;
#elif defined(FLAGGEMS_USE_NPU) || defined(FLAGGEMS_USE_MUSA)
    auto dispath_key = c10::DispatchKey::PrivateUse1;
#else
#error \
    "No backend defined. Define one of: FLAGGEMS_USE_CUDA, FLAGGEMS_USE_IX, FLAGGEMS_USE_NPU, FLAGGEMS_USE_MUSA"
#endif
    aten_lib = std::make_unique<torch::Library>(torch::Library::IMPL,
                                                std::string("aten"),
                                                c10::make_optional(dispath_key),
                                                __FILE__,
                                                __LINE__);
  }
  return *aten_lib;
}

// 定义算子注册表：算子名 -> 注册函数
using RegisterFunc = std::function<void(torch::Library&)>;
static std::unordered_map<std::string, RegisterFunc>& get_op_registry() {
  // NOTE: For "addmm.out" why we use "addmm_out" as the key?
  // Because in Python, our registered lists are similar to (op_item, func), such as ("addmm.out", addmm_out).
  // When using the `unused` parameter in the `enable` API or the `include` parameter in the `only_enable`
  // API, we pass `func.__name__` instead of `op_item`. Therefore, to allow both the Python wrapper and the
  // C++ wrapper to work simultaneously, `func.__name__` is used as the key here.
  // However, I personally think that using op_item is a better approach; let's leave that as a TODO for now.
  static std::unordered_map<std::string, RegisterFunc> registry = {
      {       "addmm",[](torch::Library& m) { m.impl("addmm", TORCH_FN(flag_gems::addmm)); }                      },
      {   "addmm_out",                [](torch::Library& m) { m.impl("addmm.out", TORCH_FN(flag_gems::addmm)); }},
      {         "bmm",                        [](torch::Library& m) { m.impl("bmm", TORCH_FN(flag_gems::bmm)); }},
      {          "mm",                   [](torch::Library& m) { m.impl("mm", TORCH_FN(flag_gems::mm_tensor)); }},
      {      "mm_out",               [](torch::Library& m) { m.impl("mm.out", TORCH_FN(flag_gems::mm_tensor)); }},
      { "max_dim_max",        [](torch::Library& m) { m.impl("max.dim_max", TORCH_FN(flag_gems::max_dim_max)); }},
      {     "max_dim",                [](torch::Library& m) { m.impl("max.dim", TORCH_FN(flag_gems::max_dim)); }},
      {         "max",                        [](torch::Library& m) { m.impl("max", TORCH_FN(flag_gems::max)); }},
      {         "sum",                        [](torch::Library& m) { m.impl("sum", TORCH_FN(flag_gems::sum)); }},
      {       "zeros",                    [](torch::Library& m) { m.impl("zeros", TORCH_FN(flag_gems::zeros)); }},
      {     "to_copy",               [](torch::Library& m) { m.impl("_to_copy", TORCH_FN(flag_gems::to_copy)); }},
      {       "copy_",                    [](torch::Library& m) { m.impl("copy_", TORCH_FN(flag_gems::copy_)); }},
      {     "nonzero",                [](torch::Library& m) { m.impl("nonzero", TORCH_FN(flag_gems::nonzero)); }},

#ifdef FLAGGEMS_POINTWISE_DYNAMIC
      {         "add",          [](torch::Library& m) { m.impl("add.Tensor", TORCH_FN(flag_gems::add_tensor)); }},
      {        "add_", [](torch::Library& m) { m.impl("add_.Tensor", TORCH_FN(flag_gems::add_tensor_inplace)); }},
      {  "add_scalar",          [](torch::Library& m) { m.impl("add.Scalar", TORCH_FN(flag_gems::add_scalar)); }},
      { "add_scalar_",
       [](torch::Library& m) { m.impl("add_.Scalar", TORCH_FN(flag_gems::add_scalar_inplace)); }                },
      { "fill_scalar",        [](torch::Library& m) { m.impl("fill.Scalar", TORCH_FN(flag_gems::fill_scalar)); }},
      {"fill_scalar_",      [](torch::Library& m) { m.impl("fill_.Scalar", TORCH_FN(flag_gems::fill_scalar_)); }},
      { "fill_tensor",        [](torch::Library& m) { m.impl("fill.Tensor", TORCH_FN(flag_gems::fill_tensor)); }},
      {"fill_tensor_",      [](torch::Library& m) { m.impl("fill_.Tensor", TORCH_FN(flag_gems::fill_tensor_)); }},
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
