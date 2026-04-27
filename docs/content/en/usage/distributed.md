---
title: Distributed Environment
weight: 70
---

# Multi-GPU Deployment

In real-world LLM deployment scenarios, multi-GPU or multi-node setups are often required
to support large model sizes and high-throughput inference.
*FlagGems* supports these scenarios by accelerating operator execution across multiple GPUs.

## 1. Single-node vs multi-node usage

For **single-node deployments**, the [enablement of FlagGems](/FlagGems/usage/basic/)
is straightforward. You can import `flag_gems` and invoke `flag_gems.enable()`
at the beginning of your program or use the context manager when apprpriate.
The *FlagGems* acceleration is then enabled without requiring any additional changes
to your code.

In **multi-node deployments**, however, the simple approach above is insufficient.
Distributed inference frameworks (like vLLM) spawn multiple worker processes across nodes,
where every process must initialize `flag_gems` individually.
If the activation occurs only in the launch script on one node, worker processes
on other nodes will fall back to the default implementations which are not accelerated.

## 2. Example: integration with vLLM and DeepSeek

To enable *FlagGems* in a distributed vLLM + DeepSeek deployment:

{{% steps %}}

1. **Baseline verification**

   Before conducting this experiment, please verify that the model can load
   and serve correctly without integrating *FlagGems*.
   For example, loading a model like `Deepseek-R1` typically requires **at least two H100 GPUs**
   and it can take **up to 20 minutes** to initialize, depending on the checkpoint size and
   the system I/O bandwidth and latency.

1. **Inject `flag_gems` into vLLM worker code**

   Locate the appropriate model runner script depending on your vLLM version:

   - If you are using the **vLLM v1 architecture** (available in vLLM ≥ 0.8),
     modify `vllm/v1/worker/gpu_model_runner.py`
   - If you are using the **legacy v0 architecture**, modify `vllm/worker/model_runner.py`

   In either file, insert the following logic after the last `import` statement:

   ```python
   import os
   if os.getenv("USE_FLAGGEMS", "false").lower() in ("1", "true", "yes"):
       try:
           import flag_gems
           flag_gems.enable()
           flag_gems.apply_gems_patches_to_vllm(verbose=True)
           logger.info("Successfully enabled flag_gems as default ops implementation.")
       except ImportError:
           logger.warning("Failed to import 'flag_gems', falling back to default implementation.")
       except Exception as e:
           logger.warning(f"Failed to enable 'flag_gems': {e}, falling back to default implementation.")
   ```

1. **Set environment variables on all nodes**

   Before launching the service, ensure the following environment variable is set
   on all nodes:

   ```shell
   export USE_FLAGGEMS=1
   ```

1. **Start distributed inference and verify**

   Launch the service and check the startup logs on each node for messages
   indicating that operators have been overridden.

   ```none
   Overriding a previously registered kernel for the same operator and the same dispatch key
   operator: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
     registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
   dispatch key: CUDA
   previous kernel: registered at /pytorch/aten/src/ATen/....
        new kernel: registered at /dev/null:488 (Triggered internally at ....)
   self.m.impl(
   ```

   This confirms that `flag_gems` has been successfully enabled across all nodes.

{{% /steps %}}
