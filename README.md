# Arbitrary Entropy Policy Optimization (AEPO)

**Entropy Is Controllable in Reinforcement Finetuning**

[![Paper](https://img.shields.io/badge/Paper-PDF-blue)]([[2510.08141] Arbitrary Entropy Policy Optimization: Entropy Is Controllable in Reinforcement Fine-tuning](https://arxiv.org/abs/2510.08141))
[![License](https://img.shields.io/badge/License-MIT-green)](./LICENSE)

---

## 🧠 Introduction

**Arbitrary Entropy Policy Optimization (AEPO)** is a novel reinforcement finetuning (RFT) framework designed to address *entropy collapse* — a critical issue in large language model (LLM) post-training where exploration vanishes as entropy monotonically decreases.

Unlike traditional entropy regularization methods that trade exploration for stability, **AEPO achieves controllable and stable entropy regulation** through three unified design principles:

1. **Policy Gradient as Regularization**
   Replaces explicit entropy bonuses with a REINFORCE-style policy gradient term applied to temperature-adjusted samples.
2. **Distribution as Regularization**
   Controls entropy implicitly by sampling from *temperature-regulated* distributions (`T_high` and `T_low`) based on current entropy state.
3. **REINFORCE as Regularization**
   Uses verifiable rewards to form one-sided, unbiased gradients that guide policy toward higher-quality distributions.

---

## 🚀 Highlights

- **Controllable Entropy:**
  Precisely maintains entropy at any arbitrary target level `H`, completely eliminating entropy collapse.
- **Entropy–Exploration–Performance Relation:**
  Demonstrates a *non-monotonic* relationship — performance improves with moderate entropy but declines when entropy becomes excessive.
- **Generalizable Framework:**
  Beyond entropy control, AEPO provides a paradigm for learning under target distributions, generalizing to multimodal alignment and reasoning tasks.

---

## 📊 Experimental Results

Experiments were conducted on **Qwen2.5-Math-7B** using the **EasyR1** and **VeRL** frameworks, across seven reasoning benchmarks:
`AIME24`, `AMC`, `College Math`, `GSM8K`, `MATH`, `Minerva`, `Olympiad`.


| Model             | Avg. Score | Entropy Stability |
| :---------------- | :--------: | :---------------: |
| Qwen2.5-Math-7B   |   37.66   |        —        |
| GRPO              |   57.96   |    ❌ Collapse    |
| Entropy-Reg       |   57.39   |   ⚠️ Unstable   |
| Entropy-Adv       |   58.18   |    ❌ Collapse    |
| **AEPO (H=0.75)** | **61.36** |     ✅ Stable     |

> 📈 AEPO outperforms all entropy-based baselines and achieves stable exploration at arbitrary entropy targets.

---

## 🧩 Method Overview

AEPO integrates into standard GRPO-style RFT pipelines as follows:

```math
\nabla_\theta J_{AEPO} =
\nabla_\theta J_{GRPO} +
\alpha \cdot \mathbb{E}_{q,o \sim \pi_\theta^T}
\left[\nabla_\theta \log \pi_\theta(o|q) \cdot R(q,o)\right],
```

where the temperature `T` switches adaptively according to the current entropy state of the policy:

```math
T =
\begin{cases}
T_{high}, & \text{if } H(\pi_{\theta_{old}}) < H_{target} \\
T_{low}, & \text{otherwise}
\end{cases}
```

## Requirements

### Software

Install via pip:

```bash
conda create -n AEPO python=3.11
conda activate AEPO
git clone https://github.com/597358816/AEPO.git
cd AEPO
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 vllm==0.8.3 transformers==4.51.2 
pip install ray==2.48.0 tensordict==0.9.1 pydantic==2.11.7
pip install flash-attn
pip install -e .
pip install tensorboard
cd example
bash qwen-math-7b-AEPO.sh
```

### Evaluation

The eval code is in：[https://github.com/QwenLM/Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math).
