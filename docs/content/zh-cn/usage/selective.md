---
title: 选择性启用算子
weight: 30
---

<!--
# Selective Operator Enablement
-->
# 选择性启用算子

<!--
When enabling *FlagGems*, you can use several optional parameters for
better control over how the operator acceleration is applied in your application.
This allows for more flexible integration and easier debugging or profiling
when working with complex workflows.
-->
在启用 *FlagGems* 库时，你可以使用若干可选的参数来精细控制在你的应用中如何使用算子加速。
这些参数的存在使得用户能够很灵活地实现各种集成任务，并且在工作流很复杂的情况下，
也可以很方便地进行故障调试和性能分析。

<!--TODO(Qiming): The `enable` and `only_enable` interfaces can be merged.-->
<!--TODO(Qiming): Add contents about operator selection using config file.-->

<!--
Currently, *FlagGems* provides three ways for you to selectively enable or disable
certain operators.
-->
目前，*FlagGems* 提供三种方式供你有选择地启用或者禁用某些算子。

<!--
- The `flag_gems.enable()` API accepts an `unused` parameter, among others.

  | Parameter      | Type        | Description                         |
  | -------------- | ----------- | ----------------------------------- |
  | `unused`       | `List[str]` | Disable specific operators          |
-->
- API 接口 `flag_gems.enable()` 可以接受一个 `unused` 参数。

  | 参数名称       | 数据类型    | 描述                    |
  | -------------- | ----------- | ------------------------|
  | `unused`       | `List[str]` | 禁用特定的算子          |

  <!--
  With this parameter, you can selectively opt out some operators which are known
  to perform not so well on your platform. For example, the following code enables
  all the supported operators except for `sum` and `add`. In other words, the listed
  operators won't be accelerated by *FlagGems*. When invoked, you will be actually
  using the PyTorch native implementation for these operators.
  -->
  使用这个参数，你可以有选择地禁用某些算子，尤其是某些算子在你的平台上表现不及预期时。
  例如，下面的代码会启用 `sum` 和 `add` 之外的所有算子。换言之，参数所列出的算子不会被
  *FlagGems* 加速。当应用调用到这些算子时，会自动回退到 PyTorch 原生的算子实现。

  ```python
  flag_gems.enable(unused=["sum", "add"])
  ```

<!--
- The `flag_gems.only_enable()` API accepts an `include` parameter, as shown below.

  | Parameter      | Type        | Description                           |
  | -------------- | ----------- | ------------------------------------- |
  | `include`      | `List[str]` | Explicitly enable specific operators. |
-->
- 接口 `flag_gems.only_enable()` 可以接受一个 `include` 参数，如下所示。

  | 参数名称       | 数据类型    | 描述                   |
  | -------------- | ----------- | -----------------------|
  | `include`      | `List[str]` | 显式地启用指定的算子   |

  <!--
  When this parameter is specified, only the listed operators will be registered
  in *FlagGems* for acceleration. All other operators will be using the PyTorch
  native implementations.
  -->
  当指定了 `include` 参数时，只有参数值中所列出的算子会在 *FlagGems*
  中被注册以执行加速版本。所有其他算子都会使用 PyTorch 原生的实现。

  ```python
  flag_gems.only_enable(include=["rms_norm", "softmax"])
  ```

  <!--
  > [!WARNING]
  > **Warning**
  >
  > The `only_enable()` interface is experimental and is subject to change
  > in future releases. Please use it with caution.
  -->
  > [!WARNING]
  > **警告**
  >
  > API 接口 `only_enable()` 是实验性质的，可能在未来版本中被移除。
  > 请谨慎使用。

<!--
- There is yet another way to perform selective operator enablement which is
  to use `use_gems()` context manager. The `use_gems()` context manager supports
  two parameters as listed below, for selective operator enablement.

  | Parameter      | Type        | Description                            |
  | -------------- | ----------- | -------------------------------------- |
  | `include`      | `List[str]` | Explicitly enable specific operators.  |
  | `exclude`      | `List[str]` | Explicitly disable specific operators. |
-->
- 除此之外，还有另外一种方式来选择性地启用算子，那就是使用 `use_gems()`
  上下文管理器。`use_gems()` 上下文管理器支持下面所列的两个参数，
  用来选择性地启用、禁用算子。

  | 参数名称       | 数据类型    | 描述                   |
  | -------------- | ----------- | -----------------------|
  | `include`      | `List[str]` | 显式地启用指定的算子   |
  | `exclude`      | `List[str]` | 显式地禁用指定的算子   |

  <!--
  The `include` parameter, when specified, behaves indentically to that of
  the `only_enable(include=...)` interface. Similarly, the `exclude` parameter,
  when specified, behaves identically to that of the `enable(unused=...)` interface.
  For example, the following code only enable the *FlagGems* acceleration
  for the operators `sum` and `and`:
  -->
  如果设置了 `include` 参数，其行为与 `only_enable(include=...)` 接口的行为完全相同。
  类似的，如果设置了 `exclude` 参数，其行为与 `enable(unused=...)` 接口的行为一致。
  例如，下面的代码仅启用 *FlagGems* 中对 `sum` 和 `and` 算子的加速：

  ```python
  # 仅在给定范围内启用指定的算子
  with flag_gems.use_gems(include=["sum", "add"]):
      # 只有 sum 和 add 会使用加速版本
      ...
  ```

  <!--
  The following code enables the *FlagGems* acceleration for **ALL** operators
  except for the `mul` and `div` operators.
  -->
  下面的代码会启用 *FlagGems* 中**所有**的加速算子，除了要排除的 `mul` 和 `div` 之外：

  ```python
  with flag_gems.use_gems(exclude=["mul", "div"]):
      # mul 和 div 算子之外的所有算子都会被加速
      ...
  ```

  <!--
  > [!TIP]
  > **Tips**
  >
  > The `include` parameter has higher priority than `exclude`.
  > If both `include` and `exclude` are provided, `exclude` is ignored.
  -->
  > [!TIP]
  > **提示**
  >
  > 参数 `include` 的优先级高于参数 `exclude`。
  > 如果两个参数都被指定，则 `exclude` 参数的设置会被忽略。
