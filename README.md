# 🧠 LLM Architecture Gallery — Study Notes & Reference Guide

> A beautifully organized, beginner-to-expert reference for understanding **modern Large Language Model architectures** — curated from [Sebastian Raschka's LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/).

[![Last Updated](https://img.shields.io/badge/Last%20Updated-June%202026-blue?style=flat-square)](https://sebastianraschka.com/llm-architecture-gallery/changelog/)
[![Models Covered](https://img.shields.io/badge/Models%20Covered-72+-green?style=flat-square)](#model-index)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/YOUR_USERNAME/llm-architecture-gallery-notes?style=flat-square)](https://github.com/YOUR_USERNAME/llm-architecture-gallery-notes)

---

## 🌟 What Is This?

This repository is a **structured, easy-to-learn companion** to Sebastian Raschka's legendary LLM Architecture Gallery. It organizes architecture diagrams, fact sheets, attention types, and key design decisions across **72+ state-of-the-art language models** — from GPT-2 all the way to trillion-parameter MoE giants.

Whether you're a student, researcher, or engineer — this is your **one-stop visual map** of how modern LLMs are built.

---

## 📚 Table of Contents

- [🗺️ Why LLM Architecture Matters](#-why-llm-architecture-matters)
- [⚡ Quick Concepts Glossary](#-quick-concepts-glossary)
- [🗂️ Model Index by Category](#-model-index-by-category)
  - [Dense Decoder Models](#-dense-decoder-models)
  - [Sparse MoE Models](#-sparse-moe-models)
  - [Hybrid Attention Models](#-hybrid-attention-models)
- [📊 Architecture Comparison at a Glance](#-architecture-comparison-at-a-glance)
- [🔬 Key Architecture Concepts Explained](#-key-architecture-concepts-explained)
- [📈 Intelligence Index Scores](#-intelligence-index-scores)
- [🔗 Resources & Further Learning](#-resources--further-learning)
- [🤝 Contributing](#-contributing)

---
# Preview: 
<img width="3276" height="2808" alt="68747470733a2f2f73656261737469616e72617363686b612e636f6d2f6c6c6d2d6172636869746563747572652d67616c6c6572792f696d616765732f6865726f2f6172636869746563747572652d67616c6c6572792d6865726f2e77656270" src="https://github.com/user-attachments/assets/e3316e15-0751-45eb-8b0f-7c134d298baa" />

## 🗺️ Why LLM Architecture Matters

Understanding **how** a model is built tells you *why* it behaves the way it does:

| Question | Architecture Clue |
|---|---|
| Why is this model fast at inference? | Sparse MoE → only a fraction of parameters activate per token |
| Why does this model handle long docs? | Sliding-window attention or linear attention |
| Why is the KV cache so small? | MLA (Multi-head Latent Attention) compresses the cache |
| Why does this model train stably? | QK-Norm or post-norm layout |

---

## ⚡ Quick Concepts Glossary

> New to LLM internals? Start here.

| Term | Plain English |
|---|---|
| **MHA** | Multi-Head Attention — the classic transformer attention, used in GPT-2 |
| **GQA** | Grouped-Query Attention — fewer KV heads → smaller cache, faster serving |
| **MQA** | Multi-Query Attention — extreme version of GQA (1 KV head) |
| **MLA** | Multi-head Latent Attention — DeepSeek's innovation to radically shrink KV cache |
| **MoE** | Mixture of Experts — only a small % of parameters "activate" per token |
| **SWA** | Sliding-Window Attention — each token only attends to its nearby neighbors |
| **RoPE** | Rotary Positional Embedding — current standard for positional encoding |
| **NoPE** | No Positional Encoding — some layers skip positional info entirely |
| **RMSNorm** | Simpler, faster normalization replacing LayerNorm |
| **QK-Norm** | Normalizing query/key vectors for training stability |
| **MTP** | Multi-Token Prediction — predicts multiple future tokens at once |
| **DeltaNet** | Linear attention alternative replacing full softmax attention |
| **Mamba-2** | State-space model layer used in NVIDIA Nemotron hybrid models |

---

## 🗂️ Model Index by Category

### 🟦 Dense Decoder Models

> All parameters active for every token. Simpler but more memory-intensive.

| Model | Scale | Context | Attention | Key Feature | Date |
|---|---|---|---|---|---|
| **GPT-2 XL** | 1.5B | 1,024 | MHA + absolute pos | Classic baseline | 2019-11 |
| **Llama 3 (8B)** | 8B | 8,192 | GQA + RoPE | Reference modern stack | 2024-04 |
| **Llama 3.2 (1B)** | 1B | 128k | GQA | Wider than Qwen3 0.6B | 2024-09 |
| **Llama 3.2 (3B)** | 3B | 128k | GQA | Tied embeddings | 2024-09 |
| **OLMo 2 (7B)** | 7B | 4,096 | MHA + QK-Norm | Post-norm for stability | 2024-11 |
| **Gemma 3 (270M)** | 270M | 128k | MQA + SWA | Tiny, 262k vocab | 2025-08 |
| **Gemma 3 (27B)** | 27B | 128k | GQA + SWA | 5:1 local/global ratio | 2025-03 |
| **Mistral Small 3.1 (24B)** | 24B | 128k | GQA | No sliding window | 2025-03 |
| **Qwen3 (0.6B)** | 0.6B | 32k | GQA | Great for local experiments | 2025-04 |
| **Qwen3 (4B / 8B / 32B)** | 4–32B | 32k–128k | GQA + QK-Norm | Reference Qwen stack | 2025-04 |
| **SmolLM3 (3B)** | 3B | 131k | GQA + NoPE layers | Every 4th layer skips RoPE | 2025-06 |
| **OLMo 3 (7B / 32B)** | 7–32B | 65k | MHA/GQA + SWA | Post-norm, transparent | 2025-11 |
| **Phi-4 (14B)** | 14B | — | — | Microsoft small model | — |
| **Gemma 4 (31B)** | 30.7B | 256k | GQA + SWA + QK-Norm | 256k multimodal | 2026-04 |
| **Nanbeige 4.1 (3B)** | 3B | 262k | GQA | On-device oriented | 2026-02 |
| **Tiny Aya (3.35B)** | 3.35B | 8,192 | GQA + SWA + NoPE | Parallel attn+MLP block | 2026-02 |

---

### 🟧 Sparse MoE Models

> Most parameters are inactive per token — efficient at scale.

| Model | Total / Active | Context | Attention | Routing Style | Date |
|---|---|---|---|---|---|
| **DeepSeek V3 (671B)** | 671B / 37B | 128k | MLA | Dense prefix + shared expert | 2024-12 |
| **DeepSeek R1 (671B)** | 671B / 37B | 128k | MLA | Same as V3, reasoning-tuned | 2025-01 |
| **DeepSeek V3.2 (671B)** | 671B / 37B | 128k | MLA + Sparse Attn | Adds sparse attention | 2025-12 |
| **Llama 4 Maverick (400B)** | 400B / 17B | 1M | GQA | Alternating dense+MoE | 2025-04 |
| **Qwen3 (235B-A22B)** | 235B / 22B | 128k | GQA + QK-Norm | No shared expert | 2025-04 |
| **Qwen3 (30B-A3B)** | 30B / 3B | 128k | GQA | Deeper, narrower | 2025-04 |
| **Qwen3 Coder Flash (30B-A3B)** | 30B / 3.3B | 256k | GQA | 128 experts, coding-tuned | 2025-07 |
| **Kimi K2 (1T)** | 1T / 32B | 128k | MLA | DeepSeek V3 scaled up | 2025-07 |
| **Kimi K2.5 (1T)** | 1T / 32B | 256k | MLA | Multimodal, 256k context | 2026-01 |
| **GLM-4.5 (355B)** | 355B / 32B | 128k | GQA + QK-Norm | 3 dense prefix layers + MTP | 2025-07 |
| **GLM-4.7 (355B)** | 355B / 32B | 202k | GQA + QK-Norm | Pre-MLA GLM baseline | 2025-12 |
| **GLM-5 (744B)** | 744B / 40B | 202k | MLA + Sparse Attn | Huge, agent-optimized | 2026-02 |
| **GPT-OSS (20B)** | 21B / 3.6B | 128k | GQA + SWA | OpenAI's open-weight | 2025-08 |
| **GPT-OSS (120B)** | 117B / 5.1B | 128k | GQA + SWA | Scaled GPT-OSS variant | 2025-08 |
| **Grok 2.5 (270B)** | 270B | 131k | GQA | Older MoE style, SwiGLU path | 2025-08 |
| **MiniMax M2 (230B)** | 230B / 10B | 196k | GQA + QK-Norm + partial RoPE | Per-layer QK-Norm | 2025-10 |
| **MiniMax-M2.5 (230B)** | 230B / 10B | 196k | GQA + QK-Norm | No hybrid attn | 2026-02 |
| **Mistral Large 3 (673B)** | 673B / 41B | 262k | MLA | Near-clone of DeepSeek V3 | 2025-12 |
| **Gemma 4 (26B-A4B)** | 25B / 3.8B | 256k | GQA + SWA | 128 experts, 8 routed | 2026-04 |
| **Arcee AI Trinity Large (400B)** | 400B / 13B | 512k | GQA + SWA + gated attn | QK-Norm + RoPE+NoPE | 2026-01 |
| **Step 3.5 Flash (196B)** | 196B / 11B | 262k | GQA + SWA | MTP-3 for high throughput | 2026-02 |
| **Xiaomi MiMo-V2-Flash (309B)** | 309B / 15B | 262k | 5:1 SWA | 128-token local window | 2025-12 |
| **Ling 2.5 (1T)** | 1T | — | — | Trillion-parameter scale | — |

---

### 🟥 Hybrid Attention Models

> Mix of attention types (e.g., Transformer + Mamba / LinearAttn). Cutting edge in 2025–2026.

| Model | Total / Active | Context | Layer Mix | Innovation |
|---|---|---|---|---|
| **Nemotron 3 Nano (30B-A3B)** | 30B / 3B | 1M | 6 GQA + 23 Mamba-2 + 23 MoE | Most extreme hybrid in gallery |
| **Nemotron 3 Super (120B-A12B)** | 120B / 12B | 1M | 8 GQA + 40 Mamba-2 + 40 MoE | Latent-space MoE + MTP |
| **Qwen3 Next (80B-A3B)** | 80B / 3B | 262k | 12 gated attn + 36 DeltaNet | DeltaNet replaces most attention |
| **Kimi Linear (48B-A3B)** | 48B / 3B | 1M | 7 MLA + 20 Kimi Delta Attn | Linear attn for 1M context |
| **xLSTM (7B)** | 7B | — | LSTM-based | Non-transformer backbone |

---

## 📊 Architecture Comparison at a Glance

```
YEAR        MODEL                   TYPE        ACTIVE    CONTEXT     ATTENTION
────────────────────────────────────────────────────────────────────────────────
2019        GPT-2 XL                Dense       1.5B      1k          MHA
2024-04     Llama 3 (8B)            Dense       8B        8k          GQA
2024-12     DeepSeek V3 (671B)      Sparse MoE  37B       128k        MLA
2025-01     DeepSeek R1 (671B)      Sparse MoE  37B       128k        MLA
2025-03     Gemma 3 (27B)           Dense       27B       128k        GQA+SWA
2025-04     Llama 4 Maverick        Sparse MoE  17B       1M          GQA
2025-04     Qwen3 (235B)            Sparse MoE  22B       128k        GQA+QK-Norm
2025-07     Kimi K2 (1T)            Sparse MoE  32B       128k        MLA
2025-08     GPT-OSS (20B / 120B)    Sparse MoE  3.6–5.1B  128k        GQA+SWA
2025-09     Qwen3 Next (80B)        Hybrid      3B        262k        DeltaNet
2025-10     Kimi Linear (48B)       Hybrid      3B        1M          Linear
2025-12     DeepSeek V3.2 (671B)    Sparse MoE  37B       128k        MLA+Sparse
2026-02     GLM-5 (744B)            Sparse MoE  40B       202k        MLA+Sparse
2026-04     Gemma 4 (31B)           Dense       31B       256k        GQA+SWA
```

---

## 🔬 Key Architecture Concepts Explained

### 🔷 Attention Mechanisms — Evolution

```
MHA (GPT-2)  →  GQA (Llama 3)  →  MLA (DeepSeek)
     ↓                ↓                  ↓
 Full KV cache   Reduced KV cache   Compressed latent KV
 300 KiB/token   128 KiB/token       68.6 KiB/token
```

**Why it matters:** KV cache size directly determines how many users you can serve simultaneously. Smaller = more efficient deployment.

---

### 🔷 Dense vs. Sparse MoE

```
Dense Model                     Sparse MoE (e.g., DeepSeek V3)
─────────────────               ─────────────────────────────────
All 8B params activate          671B total, but only 37B activate
per token                       per token (5.5% active)

Pros: simpler, faster           Pros: huge capacity, efficient
Cons: expensive to scale        Cons: complex routing, memory
```

---

### 🔷 Sliding Window vs. Global Attention

```
Global Attention:  [Token 1] ←→ [Token 1000]   (every token sees every token)
Sliding Window:    [Token 500] ←→ [Token 490–510]  (local neighborhood only)

Most models (Gemma, OLMo 3, GPT-OSS) use a ratio like:
    5 sliding-window layers : 1 global layer
```

**Why:** Long contexts become computationally cheap with SWA. Global layers preserve long-range understanding.

---

### 🔷 The Rise of Hybrid Models (2025–2026)

The newest frontier: **mix traditional transformers with linear/state-space layers.**

```
Qwen3 Next:   3:1 ratio of DeltaNet : Gated Attention
Kimi Linear:  ~3:1 ratio of Linear Attn : MLA
Nemotron:     Mamba-2 everywhere, attention only rarely
```

These models target **1M+ token contexts** with dramatically lower memory.

---

## 📈 Intelligence Index Scores

> AA Intelligence Index from [Artificial Analysis](https://artificialanalysis.ai/) — higher is better.

| Model | Total | General | Scientific | Coding | Agents |
|---|---|---|---|---|---|
| MiniMax-M2.5 | **42.0** | — | — | — | — |
| GLM-5 | 40.6 | 42.8 | 20.2 | 39.0 | 60.3 |
| Step 3.5 Flash | 38.5 | 38.5 | 32.5 | 34.6 | 48.2 |
| Kimi K2.5 | 37.3 | 44.4 | 26.0 | 25.8 | 52.8 |
| Nemotron 3 Super | 36.0 | 42.1 | 30.4 | 31.2 | 40.2 |
| MiniMax M2 | 36.0 | — | — | — | — |
| GLM-4.7 | 34.2 | 30.6 | 19.7 | 32.0 | **54.3** |
| GPT-OSS 120B | 33.3 | 37.5 | 29.1 | 28.6 | 37.9 |
| DeepSeek V3.2 | 32.1 | 29.7 | 24.2 | 34.6 | 39.8 |
| Arcee Trinity Large | 31.9 | 31.4 | 26.4 | 27.2 | 42.6 |
| Xiaomi MiMo-V2-Flash | 30.4 | 27.8 | 20.4 | 25.8 | **47.3** |
| Gemma 4 (31B) | 32.3 | 31.1 | 24.8 | 33.9 | 39.4 |
| DeepSeek R1 | 18.8 | 33.1 | 22.5 | 15.9 | 3.8 |
| Llama 4 Maverick | 18.0 | — | — | — | — |
| Qwen3 (235B-A22B) | 17.0 | 16.9 | 17.7 | 14.0 | 19.2 |
| Kimi K2 | 26.3 | 36.3 | 22.6 | 22.1 | 24.3 |
| GPT-OSS 20B | 24.5 | 29.3 | 22.5 | 18.5 | 27.6 |

---

## 🔗 Resources & Further Learning

| Resource | Description |
|---|---|
| 🌐 [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) | The original gallery — browse all 72+ model cards |
| 📰 [Ahead of AI Newsletter](https://magazine.sebastianraschka.com) | Deep-dive articles by Sebastian Raschka |
| 📖 [LLMs From Scratch (Book)](https://sebastianraschka.com/llms-from-scratch/) | Build GPT-2, Llama, Qwen3 from scratch |
| 💻 [LLMs-from-scratch GitHub](https://github.com/rasbt/LLMs-from-scratch) | Code implementations for many models |
| 📊 [Artificial Analysis](https://artificialanalysis.ai/) | Intelligence Index benchmark scores |
| 🤗 [Hugging Face Hub](https://huggingface.co/) | Model weights & configs for every model listed |

### Concept Deep Dives
- [Multi-Head Attention (MHA)](https://sebastianraschka.com/llm-architecture-gallery/mha/)
- [Grouped Query Attention (GQA)](https://sebastianraschka.com/llm-architecture-gallery/gqa/)
- [Multi-head Latent Attention (MLA)](https://sebastianraschka.com/llm-architecture-gallery/mla/)
- [Mixture of Experts (MoE)](https://sebastianraschka.com/llm-architecture-gallery/moe/)
- [Sliding Window Attention (SWA)](https://sebastianraschka.com/llm-architecture-gallery/swa/)
- [Multi-Token Prediction (MTP)](https://sebastianraschka.com/llm-architecture-gallery/mtp/)
- [RMSNorm](https://sebastianraschka.com/llms-from-scratch/ch04/09_rmsnorm/)
- [QK-Norm](https://sebastianraschka.com/llm-architecture-gallery/qk-norm/)

---

## 🏷️ Repo Name Suggestions (SEO-Optimized)

> Pick the one that fits your GitHub project best:

| Repo Name | Best For |
|---|---|
| `llm-architecture-guide` | General audience, high search volume |
| `modern-llm-architectures` | Clean, professional, discoverable |
| `llm-architecture-cheatsheet` | Students & quick reference seekers |
| `transformer-architecture-notes` | Broad keyword coverage |
| `llm-model-comparison-2025` | Year-specific SEO boost |

**Recommended:** `modern-llm-architectures` — clean, keyword-rich, professional.

---

## 🤝 Contributing

Contributions are welcome! If you:
- Spot an error in a model card summary
- Want to add notes for a new model
- Have a clearer explanation of a concept

Please open an issue or submit a pull request.

---

## 📌 About This Project

This repo was created as a **learning companion** for anyone studying modern LLM design. All original architecture diagrams and fact sheets belong to [Sebastian Raschka](https://sebastianraschka.com/) and are used here for educational reference only.

---

<!-- HASHTAGS FOR GITHUB TOPICS -->
<!-- Set these as your repo topics in GitHub Settings -->
<!-- llm, large-language-models, transformer, architecture, deep-learning, -->
<!-- machine-learning, gpt, llama, deepseek, mixture-of-experts, attention-mechanism, -->
<!-- nlp, ai, pytorch, model-comparison, study-notes, reference-guide -->

<div align="center">

**⭐ Star this repo if it helped you learn!**

Made with ❤️ for the AI learning community

</div>
