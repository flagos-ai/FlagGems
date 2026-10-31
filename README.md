[<img width="2182" height="602" alt="github+banner-20260130" src="https://github.com/flagos-ai/FlagGems/blob/master/.github/assets/banner-20260130.png" />](https://flagos.io/)

[中文版](./README_cn.md) | English

<div align="right">
  <a href="https://www.linkedin.com/company/flagos-community" target="_blank">
    <img src="https://github.com/flagos-ai/FlagGems/blob/master/docs/assets/Linkedin.png" alt="LinkIn" width="32" height="32" />
  </a>

  <a href="https://www.youtube.com/@FlagOS_Official" target="_blank">
    <img src="https://github.com/flagos-ai/FlagGems/blob/master/docs/assets/youtube.png" alt="YouTube" width="32" height="32" />
  </a>

  <a href="https://x.com/FlagOS_Official" target="_blank">
    <img src="https://github.com/flagos-ai/FlagGems/blob/master/docs/assets/x.png" alt="X" width="32" height="32" />
  </a>

  <a href="https://www.facebook.com/flagosglobalcommunity/" target="_blank">
    <img src="https://github.com/flagos-ai/FlagGems/blob/master/docs/assets/Facebook.png" alt="Facebook" width="32" height="32" />
  </a>

  <a href="https://discord.com/invite/ubqGuFMTNE" target="_blank">
    <img src="https://github.com/flagos-ai/FlagGems/blob/master/docs/assets/discord.png" alt="Discord" width="32" height="32" />
  </a>
</div>

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
- Multi-backend interface enabling support of diverse hardware platforms
- Over 10 supported backends
- C++ Triton function dispatcher (working in progress)

Check the [features](https://flagos-ai.github.io/FlagGems/overview/features/) documentation for more details.

## Getting Started

- Refer to the [Getting Started](https://flagos-ai.github.io/FlagGems/getting-started/) for a quick start.
- Refer to the [usage](https://flagos-ai.github.io/FlagGems/usage/) documentation for some details on using the software.

## Explicit BF16 small-M linear

`flag_gems.fused.bf16_small_m_linear(input, weight, *, device_sm, out=None)`
computes a forward-only dense projection for contiguous CUDA BF16 input
`[M, 6144]` and weight `[1024, 6144]`, `1 <= M <= 16`, on SM90. Call with
gradients disabled and pass `device_sm=90`, resolved before graph capture.
The optional `out` is contiguous BF16 `[M, 1024]` on the same device and must
have independent storage from the input and weight. Unsupported metadata raises
before launch; callers select their original linear path before invoking the API.
This explicit API does not replace aten linear or select a framework provider.

## Packed INT8 indexer primitives

`flag_gems.fused` exports `int8_indexer_quantize`, `int8_indexer_cache_store`
and `int8_indexer_paged_logits`. They operate on tensors, with no framework
objects or provider selection. The shared implementation uses Triton; no
TileLang implementation is moved into this library.

Quantization accepts finite FP16/BF16/FP32 unit-stride D128 heads whose leading
dimensions collapse to rows without a copy. It returns signed INT8 values and
FP32 power-of-two scales, using nearest-even rounding and a symmetric +/-127
range. The scale uses correctly rounded FP32 `max(amax, 1e-10) / 127`, then
rounds upward to a power of two using its exponent and mantissa bits. The floor
applies before division, so all-zero rows retain scale `2^-40`.
Cache-store consumes `[tokens,128]` keys and contiguous INT32/INT64
physical slots; negative slots are ignored. Nonnegative write slots must be
unique and smaller than cache capacity. A contiguous UINT8 cache page contains
`page_size*128` signed key bytes followed by `page_size` FP32 scales, rather
than interleaved records. The scale section must be FP32-aligned.

Paged logits accept INT8 `[tokens,heads,128]`, FP32 weights `[tokens,heads]`,
an integer page table and contiguous integer row-map/length vectors. The caller
folds query scales into the weights. The operator computes the INT32 dot,
ReLU, weighted head sum and then key scale, in that order. It returns fresh
FP32 `[tokens,max_model_len]` logits; masked positions, negative page IDs and
negative row mappings produce `-inf`. Nonnegative IDs and lengths must be
within supplied storage. Request mapping, cache capacity, write ordering,
causal lengths and TopK remain with the caller. Metadata failures raise before
launch; launch failures propagate. These are inference-only GPU APIs.

Query quantization and cache-store share this scale arithmetic. Paged-logits
math and the packed-cache layout are unchanged. The public wheel at
`b3cb254c55b52c450131ada0ba87109ca18c6c25` passed all 72 component-packet tests
on H100/SM90 with Torch 2.13.0+cu129 and Triton 3.6.0. This mixed CPU/metadata/GPU
suite covers the three INT8 APIs, strict FP16/BF16/FP32 scale boundaries and
ordinary graph replay with refreshed inputs and cache state. It does not qualify
RMSNorm/small-M on this package, framework integration or whole-model behavior.

## Sample models for testing

- Bert-base-uncased
- Llama-2-7b
- Llava-1.5-7b

## Contribution

<!--TODO(Qiming): replicate this to other repo.-->

- If you are interested in contributing to the FlagGems project, please refer to
  [contribution guide](https://flagos-ai.github.io/FlagGems/contribution/overview/).
  Any contributions would be highly appreciated.
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

The FlagGems project is licensed under the [Apache License (Version 2.0)](https://github.com/flagos-ai/FlagGems/blob/master/LICENSE).
