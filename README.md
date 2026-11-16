Dynamic Uncertainty-Aware Attribution
## Hallucination Detection via Token-Level Information Gain in RAG

**Course:** CS F429 Natural Language Processing — BITS Pilani Dubai — Semester 2 2025-26  
**Team Number:** 2  
**Track:** A — Dynamic Uncertainty-Aware Attribution  

**Team Members:**
| Name | ID |
|------|----|
| Abrar Akbar Mulla | 2023A7PS0173U |
| Sherwin Gummadi | 2023A7PS0287U |
| Prisha Bansal | 2023A7PS0273U |
| Sarvagya Jain | 2023A7PS0318U |

---
A hallucination detector for Retrieval-Augmented Generation (RAG) answers. It runs Mistral-7B twice, with and without the retrieved context, and checks token by token whether the context actually changed what the model expected to say. Answers that ignore their context are flagged as likely hallucinations. It combines token-level information gain, KL divergence, confidence dynamics and semantic entropy into one score, with no fine-tuning and no labelled data needed at inference time.

**Keywords:** hallucination detection, retrieval-augmented generation, RAG faithfulness, uncertainty estimation, information gain, KL divergence, semantic entropy, token-level attribution, Mistral-7B, RAGTruth, HaluEval, NLP.

## Method Overview

We detect hallucination in RAG systems by running Mistral-7B-v0.1 twice per sample:

- **Pass 1:** `Context + Response → Mistral-7B → logits_with`
- **Pass 2:** `Response only → Mistral-7B → logits_without`

The difference between these distributions measures how much the model uses the retrieved context. When a model hallucinates, it ignores the context — so the two distributions are similar. When it is faithful, the context shifts the distributions significantly.

Four metrics quantify this signal:

| Metric | Formula | Signal | Aggregation |
|--------|---------|--------|-------------|
| Information Gain | IG = H(without) − H(with) | Low = context ignored | mean_bot20 |
| KL Divergence | KL(P_with ∥ P_without) | Low = context ignored | mean_bot20 |
| Confidence Drop | max_prob(t) − max_prob(t+1) | High = unstable | mean_top20 |
| Semantic Entropy | Entropy over semantic clusters of n=3 generations | High = uncertain | sample-level |

Weights and directions are derived from 500 RAGTruth train samples and frozen for test — the test set is never used to set them.

**Key Results:**
| Dataset | AUROC | Samples |
|---------|-------|---------|
| RAGTruth test set | **0.6712** | 1000 |
| HaluEval zero-shot | **0.8579** | 500 |
| SOTA gap closed | **36.6%** | — |

## Background and theory

**The problem.** In RAG, a language model is given retrieved passages and asked to answer from them. A response is *faithful* if its claims are supported by the context and a *hallucination* if they are not. RAGTruth labels this at the level of spans in model-written answers, and HaluEval at the level of whole answers.

**Core idea: does the context change the model's mind?** Let $P_{\text{with}}(y_t \mid c, y_{<t})$ be the model's next-token distribution at response position *t* when the context *c* is in the prompt, and $P_{\text{without}}(y_t \mid y_{<t})$ the same distribution with no context. Both passes score the *same* response tokens (teacher forcing). If the response is grounded in the context, the context should make the model more certain about those tokens and shift its distribution. If the response is invented, the context has little effect, so the two distributions look alike. This is close in spirit to context-aware decoding, which also contrasts predictions with and without context.

**The four signals.**

| Signal | Definition | Reading |
|---|---|---|
| Information gain (IG) | $H(P_{\text{without}}) - H(P_{\text{with}})$ per token, where $H$ is Shannon entropy | The context should reduce uncertainty. Low or negative IG means it did not, so risk is higher. |
| KL divergence | $\mathrm{KL}(P_{\text{with}} \,\|\, P_{\text{without}})$ per token | Measures how far the context moved the distribution. Low KL means the context was ignored. |
| Confidence drop (CD) | $\max_v P_{\text{with}}(v)_t - \max_v P_{\text{with}}(v)_{t+1}$ | A sudden fall in the top-token probability suggests the model is unsure at that point. High CD means higher risk. |
| Semantic entropy (SE) | Entropy over meaning-clusters of several sampled answers (Kuhn et al., 2023) | Samples that disagree in meaning, not just wording, signal uncertainty. High SE means higher risk. |

IG and KL measure *attribution to the context*. CD and SE measure *the model's own uncertainty*, which is why the method is called uncertainty-aware.

**From tokens to one score.** Hallucinations usually affect a few tokens (a wrong number, a made-up name), so a plain average over the whole response would dilute them. Per-token scores are therefore aggregated over the most suspicious 20% of tokens: the lowest 20% for IG and KL, the highest 20% for CD. The sample-level metrics are combined in a weighted sum. The weight and direction of each metric are set from the 500 RAGTruth training samples and then frozen, so the test sets never influence them. In `results/train_calibration.json`, IG has the largest weight (about 0.37) and the highest single-metric training AUROC (about 0.66), while semantic entropy gets the smallest weight (about 0.01).

**Why AUROC.** AUROC measures how well the score ranks hallucinated answers above faithful ones across *all* thresholds, so it does not depend on picking a cut-off. 0.5 is random ranking and 1.0 is perfect.

**Implementation details worth knowing.**
- KL is computed on the 100 highest-probability tokens of $P_{\text{with}}$, which keeps it stable and cheap but is an approximation of full-vocabulary KL.
- Semantic entropy uses only n = 3 samples at temperature 1.2, clustered by sentence-embedding similarity (cosine > 0.85, `all-MiniLM-L6-v2`), so it can only take a few values. This is one reason it contributes little.
- Long contexts are left-truncated so that the response tokens are always kept.

**Limitations.** The signals are indirect: a response can ignore the context and still be true, or use the context and still contradict it. The base model (Mistral-7B-v0.1) is not instruction-tuned. RAGTruth AUROC of 0.67 is modest, and the HaluEval result is zero-shot with weights calibrated on RAGTruth. The verdict in `demo.py` (HIGH / MEDIUM / LOW) uses fixed heuristic cut-offs, not a calibrated probability. The method needs two forward passes plus sampling, which is slow (see compute times below).

## Repository Structure

```
dynamic-uncertainty-aware-attribution/
├── demo.py
├── README.md
├── requirements.txt
├── Track_A_NLP_Assignment_Final.ipynb
└── results/
    ├── results_train_500.json
    ├── results_test_1000.json
    ├── halu_results.json
    ├── train_calibration.json
    ├── df_test_scored.json
    └── exp3_temporal_precedence.png
```

## Setup

### System Requirements
- **Python version:** 3.10 or higher
- **GPU:** CUDA-enabled GPU with 16GB+ VRAM (tested on NVIDIA L4 and A100)
- **RAM:** 32GB+ recommended
- **Storage:** ~15GB for Mistral-7B model weights

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Dev-AbrarMulla/Dynamic-Uncertainty-Aware-Attribution
cd Dynamic-Uncertainty-Aware-Attribution
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

All required packages:

```
torch>=2.0.0
transformers>=4.35.0
sentence-transformers>=2.2.0
datasets>=2.14.0
scikit-learn>=1.3.0
scipy>=1.11.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
huggingface-hub>=0.19.0
accelerate>=0.24.0
bitsandbytes>=0.41.0
```

### Step 3 — HuggingFace Login
Mistral-7B-v0.1 requires a HuggingFace account with model access.

```bash
hf auth login
```

Enter your HuggingFace token when prompted. Your token needs read access to `mistralai/Mistral-7B-v0.1`. You can generate a token at (https://huggingface.co/settings/tokens).

---

## Running the Demo

### Exact command to run demo.py:
```bash
python demo.py \
    --context "Your retrieved context passage here." \
    --response "The generated response to evaluate here."
```

### Example — Hallucinated response:
```bash
python demo.py \
    --context "The Amazon rainforest covers approximately 5.5 million square kilometres and produces around 20 percent of the worlds oxygen." \
    --response "The Amazon rainforest spans over 7 million square kilometres and produces 30 percent of the worlds oxygen supply."
```

### Example — Faithful response:
```bash
python demo.py \
    --context "The Eiffel Tower was built between 1887 and 1889 as the entrance arch for the 1889 Worlds Fair. It stands 330 metres tall and was designed by engineer Gustave Eiffel." \
    --response "The Eiffel Tower was designed by Gustave Eiffel and built between 1887 and 1889 for the Worlds Fair. It stands 330 metres tall."
```

### Output
The script produces:
1. **Token-level table** — KL Divergence, Information Gain, and Confidence Drop per response token, with ⚠ flags on suspicious tokens (IG < -0.08)
2. **Semantic Entropy** — sample-level score from n=3 generations at temperature=1.2
3. **Aggregated metric scores** — mean_bot20 for IG/KL, mean_top20 for CD
4. **Hallucination verdict** — HIGH ⚠ / MEDIUM ? / LOW ✓ with reasoning

### Expected runtime
- Token-level metrics (IG, KL, CD): ~10-30 seconds per sample
- Semantic Entropy (n=3 generations): ~2-3 additional minutes per sample
- Total per sample: ~3 minutes on A100

---

## Reproducing Full Experiments

Open `Track_A_NLP_Assignment_Final.ipynb` in Google Colab with a GPU runtime. Run cells in order. All experiments are documented with inline comments.

Pre-computed results are available in `results/` — load them to skip the pipeline and directly reproduce all tables and plots.

**Full pipeline compute times:**
| Pipeline | Samples | GPU | Time |
|----------|---------|-----|------|
| Train | 500 | L4 | ~4 hours |
| Test | 1000 | L4 | ~12 hours |
| HaluEval | 500 | A100 | ~3 hours |

---

### Windows Notes
bitsandbytes may not work natively on Windows.
Recommended:
- Google Colab
- WSL2 Ubuntu
- Linux environment

## Academic Integrity

This project uses generative AI tools for code assistance and grammar correction only, as per BITS Pilani Gen AI Usage Guidelines (effective 1 April 2026). All analysis, interpretation, experimental design, and report writing is our own work.
