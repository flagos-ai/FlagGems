---
title: Selective Operator Enablement
weight: 30
---

# Selective Operator Enablement

When enabling *FlagGems*, you can use several optional parameters for
better control over how the operator acceleration is applied in your application.
This allows for more flexible integration and easier debugging or profiling
when working with complex workflows.

<!--TODO(Qiming): The `enable` and `only_enable` interfaces can be merged.-->

Currently, *FlagGems* provides three ways for you to selectively enable or disable
certain operators.

- The `flag_gems.enable()` API accepts an `unused` parameter, among others.

  | Parameter      | Type        | Description                         |
  | -------------- | ----------- | ----------------------------------- |
  | `unused`       | `List[str]` | Disable specific operators          |

  With this parameter, you can selectively opt out some operators which are known
  to perform not so well on your platform. For example, the following code enables
  all the supported operators except for `sum` and `add`. In other words, the listed
  operators won't be accelerated by *FlagGems*. When invoked, you will be actually
  using the PyTorch native implementation for these operators.

  ```python
  flag_gems.enable(unused=["sum", "add"])
  ```

- The `flag_gems.only_enable()` API accepts an `include` parameter, as shown below.

  | Parameter      | Type        | Description                           |
  | -------------- | ----------- | ------------------------------------- |
  | `include`      | `List[str]` | Explicitly enable specific operators. |

  When this parameter is specified, only the listed operators will be registered
  in *FlagGems* for acceleration. All other operators will be using the PyTorch
  native implementations.

  ```python
  flag_gems.only_enable(include=["rms_norm", "softmax"])
  ```

  > [!WARNING]
  > **Warning**
  >
  > The `only_enable()` interface is experimental and is subject to change
  > in future releases. Please use it with caution.

- There is yet another way to perform selective operator enablement which is
  to use `use_gems()` context manager. The `use_gems()` context manager supports
  two parameters as listed below, for selective operator enablement.

  | Parameter      | Type        | Description                            |
  | -------------- | ----------- | -------------------------------------- |
  | `include`      | `List[str]` | Explicitly enable specific operators.  |
  | `exclude`      | `List[str]` | Explicitly disable specific operators. |

  The `include` parameter, when specified, behaves indentically to that of
  the `only_enable(include=...)` interface. Similarly, the `exclude` parameter,
  when specified, behaves identically to that of the `enable(unused=...)` interface.
  For example, the following code only enable the *FlagGems* acceleration
  for the operators `sum` and `and`:


  ```python
  # Enable only specific ops in the scope
  with flag_gems.use_gems(include=["sum", "add"]):
      # Only sum and add will be accelerated
      ...
  ```

  The following code enables the *FlagGems* acceleration for **ALL** operators
  except for the `mul` and `div` operators.

  ```python
  with flag_gems.use_gems(exclude=["mul", "div"]):
      # All operators except mul and div will be accelerated
      ...
  ```

  > [!TIP]
  > **Tips**
  >
  > The `include` parameter has higher priority than `exclude`.
  > If both `include` and `exclude` are provided, `exclude` is ignored.
