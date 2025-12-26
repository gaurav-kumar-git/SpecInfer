🚀 LLM Decoding Strategies and Medusa Acceleration
---

This repository contains implementations of multiple Large Language Model (LLM) decoding strategies, ranging from standard sampling techniques to constrained decoding and speculative decoding using the Medusa framework. The project is divided into three main tasks and evaluated on Hindi-to-English translation using the Llama-2 model.

---

📖 Project Overview
---

The goal of this project is to implement and analyze how different decoding techniques impact text generation quality and inference speed.

Task 0 explores fundamental decoding strategies such as Greedy, Random Sampling, Top-k Sampling, and Nucleus Sampling.

Task 1 implements Word-Constrained Decoding, a greedy decoding approach that forces the generation of specific words using a Trie-based constraint mechanism.

Task 2 explores Medusa, a speculative decoding framework that accelerates inference by predicting multiple future tokens in parallel using multiple decoding heads.

---

🛠️ Installation
---

Ensure Python is installed and install the required dependencies:

pip install torch transformers datasets evaluate tqdm numpy jaxtyping medusa-llm

Note: A valid Hugging Face token with access to Llama-2 weights is required.

---

1️⃣ Task 0: Standard Decoding Strategies
---

This task implements standard decoding strategies for the Llama-2-7B model.

Implemented Strategies:

Greedy Decoding:
Selects the token with the highest probability at every step.

Random Sampling:
Samples tokens from the probability distribution using a temperature parameter.

Top-k Sampling:
Restricts sampling to the k most probable tokens.

Nucleus (Top-p) Sampling:
Samples from the smallest set of tokens whose cumulative probability exceeds threshold p.

Usage:
```bash
python task0.py --hf-token "YOUR_HF_TOKEN" --decoding-strategy topk --k 10 --max-output-len 50
```
```bash
python task0.py --hf-token "YOUR_HF_TOKEN" --decoding-strategy nucleus --p 0.9
```
Arguments:
``` bash
--decoding-strategy: greedy, random, topk, nucleus
--tau: temperature for random sampling
--k: value of k for Top-k sampling
--p: probability threshold for Nucleus sampling
```
---

2️⃣ Task 1: Word-Constrained Decoding
---

This task implements a Trie-based greedy decoding strategy that enforces the presence of specific words provided by an oracle.

Implementation Details:

A Trie is built from a list of target words.
At each decoding step, the generated prefix is checked against the Trie.
Invalid tokens are masked to force constrained word generation when active.

Usage:
``` bash
python task1.py --hf-token "YOUR_HF_TOKEN" --word-list "path/to/word_lists.txt" --max-output-len 50
```
---

3️⃣ Task 2: Medusa Decoding
---

This task utilizes the Medusa architecture, which adds multiple decoding heads to predict future tokens in parallel and accelerate inference.

Implemented Strategies:

Single-Head Decoding:
Baseline decoding using only the standard language model head.

Multi-Head Decoding:
Uses multiple Medusa heads and performs Beam Search over combined predictions.

Evaluation Metrics:

BLEU and ROUGE for generation quality.
RTF (Real Time Factor) for inference speed and latency.

Usage:
```bash
python task2.py --hf-token "YOUR_HF_TOKEN" --decoding-strategy multi-head --use-no-medusa-heads 2 --beam-width 2
```

Arguments:
```bash
--decoding-strategy: single-head or multi-head
--use-no-medusa-heads: number of Medusa heads (maximum 5)
--beam-width: beam width for candidate generation
```
---

📂 File Contents
---
```bash
├── task0.py
├── generate.py
├── task1.py
├── generate_constrained.py
├── task2.py
├── generate_medusa.py
├── Problem Statement.pdf
├── Setup Guide
└── README.md
└── word_lists.txt
```
---

📚 References
---

Llama 2: Touvron et al., 2023
Medusa: Cai et al., 2024
IndicTrans2: Gala et al., 2023
