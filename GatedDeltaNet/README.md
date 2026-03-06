# Gated Delta Networks: Improving Mamba2 with Delta Rule

![nvidia-deltanet-badge](https://github.com/user-attachments/assets/35b89293-29e9-4560-864d-45f702a5ddf7)

Official PyTorch implementation of [**Gated Delta Networks: Improving Mamba2 with Delta Rule (ICLR '25)**](https://arxiv.org/abs/2412.06464).

[![Star on GitHub](https://img.shields.io/github/stars/NVlabs/GatedDeltaNet.svg?style=social)](https://github.com/NVlabs/GatedDeltaNet/stargazers)

[Songlin Yang](https://sustcsonglin.github.io/),
[Jan Kautz](https://jankautz.com/) and
[Ali Hatamizadeh](https://research.nvidia.com/person/ali-hatamizadeh).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

For additional functionalities, such as varlen training and inference support, see [FLA implementation](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gated_deltanet).

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 32">
  <!-- Background rectangle -->
  <rect width="180" height="32" rx="6" fill="#1a1a1a"/>

  <!-- NVIDIA logo style -->
  <text x="10" y="21" font-family="Arial, sans-serif" font-weight="bold" font-size="14" fill="#76B900"></text>

  <!-- Divider -->
  <line x1="70" y1="8" x2="70" y2="24" stroke="#333" stroke-width="1"/>

</svg>

## 📢 Latest Updates
- `09/10/2025`: 🔥🔥 Gated DeltaNet has been integated as the linear component of [Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) !
- `02/23/2025`: 🔥 Check out the optimized [Gated DeltaNet FLA kernels](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/fla_version/) with significantly faster speed.
- `02/22/2025`: 🔥 Gated DeltaNet is available in [FLA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule) !
- `01/22/2025`: 🔥🔥 Gated DeltaNet has been accepted to ICLR '25.
- `12/09/2024`: **Code Release**: Train your own Gated DeltaNet on Slimpajama dataset
- Watch this space for more exciting updates!

---

## ❓ Frequently Asked Questions (FAQ)

### 1️⃣ Can I use Gated DeltaNet directly from FLA?

Yes! You can import the Gated DeltaNet block directly from FLA. The following script demonstrates how to do so using either FLA or our repository:

```py
>>> USE_FLA = True
>>> import torch
>>> if USE_FLA:
...     from fla.layers import GatedDeltaNet
>>> else:
...     from .gated_delta_net import GatedDeltaNet
>>>
>>> bs, num_heads, seq_len, hidden_size = 16, 4, 2048, 512
>>> gated_deltanet = GatedDeltaNet(hidden_size=hidden_size, num_heads=num_heads, mode='chunk').bfloat16().cuda()
>>> gated_deltanet
GatedDeltaNet(
  (silu): SiLU()
  (q_proj): Linear(in_features=512, out_features=1024, bias=False)
  (k_proj): Linear(in_features=512, out_features=1024, bias=False)
  (v_proj): Linear(in_features=512, out_features=2048, bias=False)
  (b_proj): Linear(in_features=512, out_features=4, bias=False)
  (a_proj): Linear(in_features=512, out_features=4, bias=False)
  (q_conv1d): ShortConvolution(1024, 1024, kernel_size=(4,), stride=(1,), padding=(3,), groups=1024, bias=False, activation=silu)
  (k_conv1d): ShortConvolution(1024, 1024, kernel_size=(4,), stride=(1,), padding=(3,), groups=1024, bias=False, activation=silu)
  (v_conv1d): ShortConvolution(2048, 2048, kernel_size=(4,), stride=(1,), padding=(3,), groups=2048, bias=False, activation=silu)
  (g_proj): Linear(in_features=512, out_features=2048, bias=False)
  (o_norm): FusedRMSNormSwishGate(512, eps=1e-05)
  (o_proj): Linear(in_features=2048, out_features=512, bias=False)
)
>>> x = torch.randn(bs, seq_len, hidden_size).bfloat16().cuda()
>>> y, _, _ = gated_deltanet(x)
>>> y.shape
torch.Size([16, 2048, 512])
```

---

### 2️⃣ What is the difference between the FLA Gated DeltaNet kernels and the NVLabs implementation?

FLA kernels are **faster** and also support **variable-length (varlen) training**. We **strongly recommend** using FLA for better performance.

For reference, we also provide FLA-based kernels in this repository. You can find the optimized FLA Gated DeltaNet kernels [here](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/fla_version/).

---

### 3️⃣ Will you release the pretrained model weights?

No, we only provide code implementations.

---

### 4️⃣ The dataloader in this repository is designed for SlimPajama-672B, but your models were trained on FineWeb-Edu. Why is that, and should I expect similar results?

For the code release, we used the original [Samba](https://github.com/microsoft/Samba) repository and included the **SlimPajama-672B** dataloader to maintain consistency.

Our experiments confirm that **SlimPajama-672B produces similar results and trends** to those reported in our paper. You can expect comparable performance.

---

### 5️⃣ Any guidance for evaluating the models?

Since this codebase is primarily adapted from the [Samba codebase](https://github.com/microsoft/Samba), which is designed mainly for training, evaluation can be inconvenient. Notably, Samba codebase lacks generation utilities required for many generation-based evaluation tasks.

We recommend first converting your trained model weights to Hugging Face format provided in the [FLA repo](https://github.com/fla-org/flash-linear-attention). Once converted, you can leverage FLA for streamlined evaluation.

- **For Single Needle in a Haystack (S-NIAH) tasks:**
  Please install [NVIDIA/RULER](https://github.com/NVIDIA/RULER/). The installation process can be challenging; we suggest installing any missing dependencies individually to ensure success. S-NIAH tasks are zero-shot tasks, and since RULER supports Hugging Face format models, you can easily evaluate your converted FLA models in this case.

- **For zero-shot commonsense reasoning tasks (Table 3):**
  Follow the [FLA instructions](https://github.com/fla-org/flash-linear-attention/tree/main?tab=readme-ov-file#evaluations) for evaluation details.

- **For zero-shot, in-context recall-intensive tasks (Table 4):**
  Use the official [evaluation script](https://github.com/HazyResearch/prefix-linear-attention/blob/main/lm-eval-harness/prompt_scripts/run_jrt_prompt_hf.sh) from their repository.
  ⚠️ **Important:** Avoid directly using `lm-eval-harness` with the task name alone, as this can lead to significant performance differences. These retrieval tasks are highly prompt-sensitive for instruction-untuned models in zero-shot settings.


## 🌟 Why Gated DeltaNet?

Gated DeltaNet introduces a novel approach to linear transformers by combining:
- 🧠 **Smart Memory Management**: Intelligent memory management that knows what to keep and what to forget
- ⚡ **Precise Updates**: Targeted memory updates that enhance model efficiency
- 💻 **Hardware Efficiency**: Optimized implementation for real-world deployment

![Architecture Overview](https://github.com/user-attachments/assets/70f96a7e-e51d-4514-a429-2ae30c52afbb)


### Efficiency
Gated DeltaNet shows exceptional performance in terms of training throughput compared to models like Mamba2 and Samba:

<p align="center">
<img src="https://github.com/user-attachments/assets/b5c96369-a998-442b-ad7c-2f9fb6979b44" width=62% height=62%
class="center">
</p>


### Language Modeling and Reasoning

Our model outperforms competitors of various types(e.g. Transformer, RNN, hybrid) in terms of perplexity and zero-shot accuracy on reasoning benchmarks:

<p align="center">
<img src="https://github.com/user-attachments/assets/afaa4527-e974-4367-a784-6e19c21c8bc0" width=82% height=82%
class="center">
</p>


### Long-context

Gated DeltaNet also achieves favorable perplexity scores on long-context benchmarks:

<p align="center">
<img src="https://github.com/user-attachments/assets/64c307f4-3b30-4899-ab17-6507e6506c51" width=72% height=72%
class="center">
</p>


## 🚀 Getting Started

### Training Your Model

Launch your training with our streamlined command:

```bash
python ../pretrain.py \
--train_data_dir ${TRAIN_DATA} \
--val_data_dir ${VALIDATION_DATA} \
--output_root ${SAVE_DIR} \
--exp_name ${NAME} \
--model_name ${MODEL} \
--train_config ${CONFIG} \
--eval_iters ${EVAL_ITERS} \
--learning_rate ${LR} \
--micro_batch_size ${MICRO_BATCH_SIZE}
```
💡 **Pro Tip**: Add `--interactive_job --debug` for interactive debugging sessions!

Please see this slurm [script](https://github.com/NVlabs/GatedDeltaNet/blob/main/scripts/tsz512x4k_15B_gated_deltanet_h1_0.4B.sh) for training the GatedDeltaNet_H1 model with 0.4B parameters on 15B tokens. The training requires 4 nodes and can be finished in approximately 4 hours. For this run, the validation loss and perplexitty curves (1x & 2x for lengh extrapolation) are expected as follows:

![curves](https://github.com/user-attachments/assets/bd8afd42-6f20-4103-8b31-48516871b681)


## 📜 License

Copyright © 2025, NVIDIA Corporation. All rights reserved.

Licensed under the NVIDIA Source Code License-NC. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

Built on the shoulders of giants:
- [Samba](https://github.com/microsoft/Samba)
- [LiTGPT](https://github.com/Lightning-AI/litgpt)
- [TinyLLaMa](https://github.com/jzhang38/TinyLlama)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)

## ⭐ Support Us

If you find this work useful, please consider:
- Starring the repository
- Citing our paper
- Contributing to the codebase

Join us in pushing the boundaries of linear transformers! 🚀

## Citation

If you find Gated DeltaNet to be useful for your work, please consider citing our paper:

```
@inproceedings{yang2025gated,
title={Gated Delta Networks: Improving Mamba2 with Delta Rule},
author={Songlin Yang and Jan Kautz and Ali Hatamizadeh},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=r8H7xhYPwz}
}
```

## Star History

[![Stargazers repo roster for @NVlabs/GatedDeltaNet](https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=NVlabs&repo=GatedDeltaNet)](https://github.com/NVlabs/GatedDeltaNet/stargazers)


[![Star History Chart](https://api.star-history.com/svg?repos=NVlabs/GatedDeltaNet&type=Date)](https://star-history.com/#NVlabs/GatedDeltaNet&Date)
