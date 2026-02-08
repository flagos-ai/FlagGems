[<img width="2182" height="602" alt="github+banner-20260130" src=".github/assets/banner-20260130.png" />](https://flagos.io/)

[[中文版](./README_cn.md)|English]

## About

FlagGems is part of [FlagOS](https://flagos.io/), a fully open-source system software stack
designed to unify the model–system–chip layers and foster an open and collaborative ecosystem.
It enables a "develop once, run anywhere" workflow across diverse AI accelerators,
unlocking hardware performance, eliminating fragmentation among AI chipset-specific software stacks,
and substantially lowering the cost of porting and maintaining AI workloads.

FlagGems is a high-performance, generic operator library implemented in [Triton](https://github.com/openai/triton) language.
It is built on a collection of backend-neutral kernels that aims to accelerate LLM (Large-Language Models) training
and inference across diverse hardware platforms.

By registering with the ATen backend of [PyTorch](https://pytorch.org/), FlagGems facilitates a seamless transition,
allowing model developers to switch to Triton without changing the low level APIs.
Users can continue using their familiar Pytorch APIs while at the same time benefit from new hardware acceleration technologies.
For kernel developers, the Triton language offers readability, user-friendliness and performance comparable to CUDA.
This convenience allows developers to engage in the development of FlagGems with minimal learning investment.

## Features

FlagGems provides the following technical features.

- A large collection of PyTorch compatible operators
- Hand-optimized performance for selective operators
- Eager-mode ready, independent of `torch.compile`
- Automatic pointwise operator codegen supporting arbitrary input types and layout
- Fast per-function runtime kernel dispatching
- Multi-backend interface enabling support of [diverse hardware platforms](./docs/features.md#generic-interface-for-diverse-platforms)
- Over 10 supported backends
- C++ Triton function dispatcher (working in progress)

Check the [features](./docs/features.md) documentation for more details.

## Getting Started

- Refer to the [Getting Started](./docs/getting-started.md) for a quick start.
- Refer to the [usage](./docs/usage.md) documentation for some details on using the software.
- Refer to the [how-to-use](./docs/how_to_use_flaggems.md) documentation for some details
  on configuration options.

## Sample models for testing

- Bert-base-uncased
- Llama-2-7b
- Llava-1.5-7b

## Contribution

<!--TODO(Qiming): replicate this to other repo.-->

- If you are interested in contributing to the FlagGems project, please refer to [CONTRIBUTING.md](./CONTRIBUTING.md). Any contributions would be highly appreciated.
- Please file an issue for feature requests or bug reports.
- Drop us an email at <a href="mailto:contact@flagos.io">contact@flagos.io</a> when you have questions or suggestions to share.
- Join the FlagGems WeChat group by scanning the QR code below. You will receive first-hand messages about updates and new releases.
  Let the team know your questions or ideas!

  <img width="204" height="180" alt="开源小助手" src="https://github.com/user-attachments/assets/da42799f-c7f7-43f0-91c3-f4935b24e968" />

## Citation

If you find our work useful, please consider citing our project:

```bibtex
@misc{flaggems2024,
    title={FlagOS/FlagGems: An operator library for large language models implemented in the Triton language.},
    url={https://github.com/flagos-ai/FlagGems},
    journal={GitHub},
    author={The FlagOS contributors},
    year={2024}
}
```

## License

The FlagGems project is licensed under the [Apache License (Version 2.0)](./LICENSE).
