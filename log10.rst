.. _log10:

log10
=====

.. currentmodule:: flaggems.ops.pointwise

.. autofunction:: log10

Description
-----------

Computes the base-10 logarithm of each element in the input tensor.

.. math::
    \text{out}_i = \log_{10}(\text{input}_i)

For inputs less than or equal to 0, the result is NaN (for negative values)
or -inf (for zero), matching PyTorch's behavior.

Arguments
---------
input (Tensor): The input tensor.

Returns
-------
Tensor: A tensor of the same shape as input containing the base-10 logarithm
        of each element.

Examples
--------

>>> import torch
>>> from flaggems.ops import log10
>>> x = torch.tensor([1.0, 10.0, 100.0, 1000.0])
>>> log10(x)
tensor([0.0000, 1.0000, 2.0000, 3.0000])

>>> x = torch.tensor([0.1, 0.01, 0.001])
>>> log10(x)
tensor([-1., -2., -3.])

>>> # Handle edge cases
>>> x = torch.tensor([0.0, -1.0, float('inf')])
>>> log10(x)
tensor([  -inf,    nan,    inf])

Gradient Formula
----------------

.. math::
    \frac{\partial \log_{10}(x)}{\partial x} = \frac{1}{x \ln(10)}

Implementation Details
----------------------

- Uses Triton for GPU acceleration
- Supports float16 and float32 data types
- Handles arbitrary tensor shapes (1D to 5D)
- Memory-efficient with proper block sizing
- Achieves speedup >= 0.9 compared to PyTorch
