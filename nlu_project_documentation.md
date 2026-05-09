# NLU Project Documentation
## Opinion Mining with LoRA Fine-Tuning
### Base Model vs. Fine-Tuned Comparison on IMDB Reviews

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Model Structure](#3-model-structure)  
   3.1 [Base Model — Qwen2.5-0.5B](#31-base-model--qwen25-05b)  
   3.2 [4-bit Quantization (BitsAndBytes)](#32-4-bit-quantization-bitsandbytes)  
   3.3 [LoRA Adapter Layer](#33-lora-adapter-layer)  
4. [Code Walkthrough](#4-code-walkthrough)  
   4.1 [Cell 1 — Installation](#41-cell-1--installation)  
   4.2 [Cell 2 — Dependencies & Environment](#42-cell-2--dependencies--environment)  
   4.3 [Cell 3 — Dataset Preparation & Feature Labeling](#43-cell-3--dataset-preparation--feature-labeling)  
   4.4 [Cell 4 — Base Model Loading](#44-cell-4--base-model-loading)  
   4.5 [Cell 5 — Evaluation Helpers](#45-cell-5--evaluation-helpers)  
   4.6 [Cell 6 — Base Model Evaluation (Before Fine-Tuning)](#46-cell-6--base-model-evaluation-before-fine-tuning)  
   4.7 [Cell 7 — LoRA Configuration](#47-cell-7--lora-configuration)  
   4.8 [Cell 8 — Training Pipeline](#48-cell-8--training-pipeline)  
   4.9 [Cell 9 — Fine-Tuned Model Evaluation](#49-cell-9--fine-tuned-model-evaluation)  
   4.10 [Cell 10 — Side-by-Side Comparison & Plots](#410-cell-10--side-by-side-comparison--plots)  
   4.11 [Cell 11 — Live Demo](#411-cell-11--live-demo)  
5. [Results & Metrics](#5-results--metrics)
6. [Design Decisions & Trade-offs](#6-design-decisions--trade-offs)
7. [Limitations](#7-limitations)

---

## 1. Project Overview

This project demonstrates **opinion mining** from text reviews using a two-stage approach:

1. **Test the base model** on a structured multi-field extraction task — no training, zero-shot.
2. **Fine-tune the same model** with LoRA adapters on labeled examples — then test again.
3. **Compare both** across 5 metrics to quantify what fine-tuning actually adds.

The task is richer than simple sentiment classification. Instead of outputting one word, the model must produce a structured string with four fields:

```
sentiment: positive | emotions: joy, excitement | aspects: acting, plot | intensity: strong
```

This structured output is intentionally harder, which makes the base model visibly struggle and the fine-tuning gap much more meaningful.

**Dataset:** IMDB movie reviews (25,000 train / 25,000 test available; we use 8,000 / 800)  
**Base model:** `Qwen/Qwen2.5-0.5B` — 500M parameter causal language model  
**Hardware target:** Kaggle Free Tier — NVIDIA T4 GPU, 16GB VRAM  
**Training time:** ~45–55 minutes  

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        IMDB Dataset                         │
│              (8,000 train  /  800 test samples)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Feature Derivation (Rule-Based)               │
│  sentiment → from IMDB label                                │
│  emotions  → keyword matching across 7 emotion categories   │
│  aspects   → keyword matching across 6 aspect categories    │
│  intensity → 4-signal scoring (words + ! + CAPS + length)   │
└──────────────────────────┬──────────────────────────────────┘
                           │  Instruction-Response pairs
                           ▼
┌──────────────────────────────────┐
│        Qwen2.5-0.5B Base Model   │
│        (4-bit NF4 quantized)     │
│        315M active parameters    │
└──────────┬───────────────────────┘
           │                    │
           │ Zero-shot          │ + LoRA Adapters (r=32)
           │ evaluation         │   17.6M trainable params
           ▼                    ▼
    ┌─────────────┐      ┌─────────────────────┐
    │  Base Model │      │  Fine-Tuned Model   │
    │  Results    │      │  Results            │
    └──────┬──────┘      └────────┬────────────┘
           │                      │
           └──────────┬───────────┘
                      ▼
           ┌─────────────────────┐
           │  5-Metric Comparison│
           │  + Visualizations   │
           └─────────────────────┘
```

---

## 3. Model Structure

### 3.1 Base Model — Qwen2.5-0.5B

**Qwen2.5-0.5B** is a decoder-only transformer (GPT-style) developed by Alibaba. It is the smallest model in the Qwen2.5 family.

| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Total parameters | ~500M (315M active after quantization overhead) |
| Context window | 32,768 tokens |
| Layers | 24 transformer blocks |
| Attention heads | 14 |
| Hidden dimension | 896 |
| Vocabulary size | 151,643 tokens |
| Training data | Multi-lingual web text |

Each transformer block contains:
- **Self-attention** with Query, Key, Value, and Output projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **Feed-forward MLP** with three linear layers (`gate_proj`, `up_proj`, `down_proj`) — uses SwiGLU activation
- **RMS Layer Normalization** before each sub-layer
- **Rotary Position Embeddings (RoPE)** for positional encoding

The model generates text **autoregressively**: it predicts the next token one at a time, conditioned on all previous tokens, until it hits the max token limit or an end-of-sequence token.

---

### 3.2 4-bit Quantization (BitsAndBytes)

Loading a 500M parameter model in full float32 precision requires ~2GB of VRAM. On a Kaggle T4 with 16GB shared, this is fine — but we also need memory for gradients, optimizer states, and activations during training. Quantization reduces the footprint significantly.

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               # Store weights in 4-bit instead of 16-bit
    bnb_4bit_use_double_quant=True,  # Quantize the quantization constants too (extra saving)
    bnb_4bit_quant_type="nf4",       # NormalFloat4: optimal for normally-distributed weights
    bnb_4bit_compute_dtype=torch.float16  # Dequantize to fp16 during computation
)
```

**How it works:**
- Weights are stored in 4-bit NF4 format (16× compression vs float32)
- At computation time, weights are dequantized on-the-fly to float16
- Only the activations (intermediate values) stay in float16 throughout
- Net result: ~75% VRAM reduction with ~1–2% accuracy loss vs full precision

**NF4 (NormalFloat4)** is specifically designed for transformer weights, which tend to follow a normal (bell-curve) distribution. It places more quantization "bins" near zero where weights cluster, giving better fidelity than uniform 4-bit quantization.

---

### 3.3 LoRA Adapter Layer

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique. Instead of updating all 500M weights (which is expensive and requires storing full-precision copies), LoRA freezes the base weights and injects small trainable matrices alongside them.

**The core idea:**

For a weight matrix `W` of shape `(d_out, d_in)`, instead of updating `W` directly, LoRA learns two small matrices:
- `A` of shape `(r, d_in)` — initialized randomly
- `B` of shape `(d_out, r)` — initialized to zero

The effective weight becomes: `W + (lora_alpha/r) × B × A`

Because `r << d_in` and `r << d_out`, the number of trainable parameters is tiny.

```python
peft_config = LoraConfig(
    r=32,           # Rank — controls adapter size. Higher = more capacity, more memory
    lora_alpha=64,  # Scaling factor. Rule of thumb: alpha = 2 × r
    target_modules=[
        # Attention layers — learn task-specific query/key/value patterns
        "q_proj", "k_proj", "v_proj", "o_proj",
        # MLP layers — learn task-specific feed-forward transformations
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,   # Regularization: randomly zero out adapter outputs during training
    bias="none",         # Don't train bias terms (minimal gain, extra cost)
    task_type="CAUSAL_LM"
)
```

**Why target both attention AND MLP?**

Earlier versions of this project only targeted attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Adding MLP layers (`gate_proj`, `up_proj`, `down_proj`) gives LoRA more surface area to adapt the model's behavior. For a new output *format* (structured multi-field extraction), the MLP layers — which handle content transformation — matter just as much as attention.

**Trainable parameter count:**

| Config | Trainable Params | % of Total |
|---|---|---|
| Full fine-tune | 500M | 100% |
| LoRA r=16, attention only | 2.2M | 0.44% |
| LoRA r=32, attention + MLP | 17.6M | 3.44% |

We train only 3.44% of parameters, yet the model learns the new task effectively.

---

## 4. Code Walkthrough

### 4.1 Cell 1 — Installation

```python
!pip install -q -U transformers datasets bitsandbytes peft trl accelerate scikit-learn matplotlib seaborn
```

| Package | Purpose |
|---|---|
| `transformers` | Model loading, tokenizer, generation |
| `datasets` | IMDB dataset loading and processing |
| `bitsandbytes` | 4-bit quantization engine |
| `peft` | LoRA adapter injection and management |
| `trl` | `SFTTrainer` — supervised fine-tuning wrapper |
| `accelerate` | Hardware abstraction (GPU/CPU dispatch) |
| `scikit-learn` | Classification reports, confusion matrices |
| `matplotlib`, `seaborn` | Plots and visualizations |

---

### 4.2 Cell 2 — Dependencies & Environment

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

This line is critical on Kaggle. Kaggle sometimes provides two GPUs, and PyTorch's `DataParallel` will try to split the model across both. But `bitsandbytes` 4-bit quantization stores metadata tied to a specific device — splitting it across GPUs causes a `RuntimeError` or `AcceleratorError`. Setting `CUDA_VISIBLE_DEVICES="0"` makes Python see only one GPU, preventing this entirely.

This must be set **before** any CUDA initialization — hence it lives in the very first code cell.

---

### 4.3 Cell 3 — Dataset Preparation & Feature Labeling

IMDB only provides binary sentiment labels (0 = negative, 1 = positive). We need 4 fields. The `derive_features()` and `derive_intensity()` functions automatically generate the missing fields from the text itself using rule-based heuristics.

#### Emotion Detection

```python
EMOTION_KEYWORDS = {
    "joy":            ["enjoy", "love", "delight", "wonderful", ...],
    "anger":          ["hate", "awful", "terrible", "rage", ...],
    "sadness":        ["sad", "cry", "tears", "heartbreak", ...],
    "excitement":     ["excit", "thrilling", "amazing", ...],
    "disappointment": ["disappoint", "boring", "dull", "waste", ...],
    "fear":           ["scary", "terrify", "horror", "nightmare", ...],
    "surprise":       ["surpris", "unexpected", "twist", "shock", ...],
}
```

Each emotion has a list of trigger keywords. The function counts keyword hits per emotion and returns the top 2. If no emotion matches, a default is assigned based on sentiment (joy for positive, disappointment for negative).

#### Aspect Detection

```python
ASPECT_KEYWORDS = {
    "acting":    ["act", "perform", "cast", "actor", "actress", ...],
    "plot":      ["plot", "story", "script", "storyline", ...],
    "visuals":   ["visual", "cinemat", "cgi", "beautifully shot", ...],
    "direction": ["direct", "director", "pacing", "edit", ...],
    "music":     ["music", "score", "soundtrack", "song", ...],
    "humor":     ["funny", "humor", "comedy", "laugh", ...],
}
```

All aspects that appear in the review are included (not just top 2). Aspect detection is binary: either the keyword appears or it doesn't. If nothing matches, `["plot"]` is the fallback.

#### Intensity Detection — 4-Signal Heuristic

```python
def derive_intensity(text):
    score = 0
    score += min(strong_hits, 3)        # Signal 1: strong/superlative words (max 3 pts)
    score += min(text.count('!'), 2)    # Signal 2: exclamation marks (max 2 pts)
    score += min(len(caps_words), 2)    # Signal 3: ALL-CAPS words ≥3 chars (max 2 pts)
    if word_count > 250: score += 2     # Signal 4: long review = more investment
    elif word_count > 100: score += 1

    if score >= 4: return "strong"
    elif score >= 2: return "moderate"
    else: return "weak"
```

**Why 4 signals?** The previous version used only word count + strong word count. This failed on short-but-emphatic reviews like `"This was GARBAGE!"` — only 4 words and 1 strong word, so it scored as "weak" despite being clearly intense. The new heuristic correctly captures ALL-CAPS shouting and exclamation marks as independent intensity signals.

#### Safe Serialization

```python
return {"text": prompt, "sentiment": sentiment,
        "emotions": json.dumps(emotions),   # stores ["joy", "anger"] as a JSON string
        "aspects":  json.dumps(aspects),
        "intensity": intensity}
```

Emotions and aspects are lists. HuggingFace datasets columns must be scalar strings, so we serialize with `json.dumps()` and deserialize later with `json.loads()`. This replaces the previous `eval()` which was a security risk.

---

### 4.4 Cell 4 — Base Model Loading

```python
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},       # Pin ALL layers to GPU 0 — no splitting
    trust_remote_code=True,
    dtype=torch.float16
)
base_model.config.use_cache = False
```

`device_map={"": 0}` means "put everything on device 0". This is different from `device_map="auto"` which distributes layers across all visible GPUs — exactly the multi-GPU bug we fixed earlier.

`use_cache=False` disables the KV-cache during training. The cache speeds up generation but is incompatible with gradient checkpointing and wastes memory during training when we process full sequences anyway.

---

### 4.5 Cell 5 — Evaluation Helpers

Three functions handle all evaluation logic:

#### `parse_output(text)`

Uses regex to extract each field from the model's raw text output:

```python
sm = re.search(r'sentiment:\s*(positive|negative)', text, re.I)
em = re.search(r'emotions?:\s*([^|\n]+)', text, re.I)
am = re.search(r'aspects?:\s*([^|\n]+)', text, re.I)
im = re.search(r'intensity:\s*(strong|moderate|weak)', text, re.I)
```

The regex is intentionally flexible (`re.I` for case-insensitive, `emotions?` for singular/plural). A response is marked `valid=True` only if at least the sentiment field was found.

#### `evaluate_model(model, tokenizer, dataset, num_samples, label)`

Loops over `num_samples` test examples. For each one:
1. Strips the ground-truth response from the formatted text to create an inference prompt
2. Runs `model.generate()` with `do_sample=False` (greedy decoding — no randomness)
3. Parses the output with `parse_output()`
4. Stores prediction vs ground truth for all 4 fields

`do_sample=False` is important for a classification task — we want the model's most confident answer, not a random sample. Temperature and sampling introduce noise that would make results non-reproducible.

#### `score_results(results, label)`

Computes 5 metrics from the collected results:

| Metric | How it's computed |
|---|---|
| **Format compliance** | % of responses where sentiment field was found |
| **Sentiment accuracy** | % of valid responses where predicted sentiment = true sentiment |
| **Emotion hit rate** | % where predicted emotions and true emotions share at least 1 label |
| **Aspect hit rate** | % where predicted aspects and true aspects share at least 1 label |
| **Intensity accuracy** | % of valid responses where predicted intensity = true intensity |

Emotion and aspect use **set intersection** (`set(pred) & set(true)`) rather than exact match, because the order of emotions/aspects doesn't matter and partial matches are meaningful.

---

### 4.6 Cell 6 — Base Model Evaluation (Before Fine-Tuning)

```python
base_results = evaluate_model(base_model, tokenizer, test_dataset, num_samples=150, label="Base")
base_scores  = score_results(base_results, label="BASE MODEL (no fine-tuning)")
```

This runs the model exactly as downloaded — no training, no task-specific adaptation. The base model has seen instruction-following data during pre-training, which is why it can produce structured output at all. But it has never been asked to extract these specific fields from reviews, so results are noticeably weaker than after fine-tuning.

**Observed results:**

```
Format compliance :  97.3%  (146/150 parseable)
Sentiment accuracy:  70.7%
Emotion hit rate  :  69.3%
Aspect hit rate   :  91.3%
Intensity accuracy:  32.0%
```

---

### 4.7 Cell 7 — LoRA Configuration

```python
base_model = prepare_model_for_kbit_training(base_model)
```

This PEFT utility does two things to the quantized model before attaching LoRA:
1. Casts all layer normalization weights to float32 (they must stay in full precision to prevent training instability)
2. Enables gradient checkpointing — instead of storing all intermediate activations in memory, it recomputes them during the backward pass, trading compute time for memory

```python
peft_config = LoraConfig(r=32, lora_alpha=64, target_modules=[...], ...)
model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
# → trainable params: 17,596,416 || all params: 511,629,184 || trainable%: 3.44%
```

`get_peft_model()` wraps the base model and inserts LoRA matrices at each targeted layer. The base weights become frozen (their `requires_grad` is set to `False`). Only the ~17.6M LoRA parameters are updated during training.

---

### 4.8 Cell 8 — Training Pipeline

```python
sft_config = SFTConfig(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,     # effective batch = 4 × 4 = 16
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_length=700,
)
```

**Key hyperparameter decisions:**

| Parameter | Value | Reason |
|---|---|---|
| `batch_size=4` | 4 | Maximum that fits in 16GB with r=32 LoRA + 4-bit |
| `gradient_accumulation=4` | 4 | Simulates batch size of 16 without extra memory |
| `paged_adamw_32bit` | — | Optimizer states stored in CPU RAM and paged to GPU as needed — saves ~4GB vs standard AdamW |
| `learning_rate=2e-4` | 2e-4 | Standard for LoRA; higher than full fine-tuning because only a small % of params are being updated |
| `cosine` scheduler | — | Gradually reduces LR to near-zero by end of training, preventing overshooting |
| `warmup_ratio=0.03` | 3% | First 3% of steps gradually increase LR from 0 — prevents instability at the start |
| `max_length=700` | 700 | Longer than v2 (512) to accommodate the structured 4-field output |

`SFTTrainer` (Supervised Fine-Tuning Trainer) from TRL handles the training loop. It is designed specifically for instruction-tuning: it reads the `dataset_text_field` column, tokenizes it, and trains the model to predict the response portion (everything after `### Response:`).

---

### 4.9 Cell 9 — Fine-Tuned Model Evaluation

```python
ft_results = evaluate_model(model, tokenizer, test_dataset, num_samples=150, label="LoRA")
ft_scores  = score_results(ft_results, label="FINE-TUNED MODEL (LoRA r=32, 3 epochs, 8k samples)")
```

Identical setup as the base model evaluation — same 150 samples, same prompt, same greedy decoding. Only the model weights differ (base weights frozen + LoRA adapters active).

**Observed results:**

```
Format compliance : 100.0%  (150/150 parseable)
Sentiment accuracy:  80.7%
Emotion hit rate  :  69.3%
Aspect hit rate   : 100.0%
Intensity accuracy:  40.7%
```

---

### 4.10 Cell 10 — Side-by-Side Comparison & Plots

Two sub-cells produce the comparison output:

**Sub-cell A — Classification reports:** Runs `sklearn.metrics.classification_report` on sentiment predictions for both models. This gives precision, recall, and F1-score broken down by class (positive/negative), not just overall accuracy.

**Sub-cell B — 6-panel figure:**

| Panel | What it shows |
|---|---|
| Top-left (spans 2 columns) | Grouped bar chart: all 5 metrics, base (red) vs LoRA (green) |
| Top-right | Gain chart: LoRA improvement over base in percentage points |
| Bottom-left | Confusion matrix for base model sentiment |
| Bottom-center | Confusion matrix for fine-tuned model sentiment |
| Bottom-right | Summary text box with all numbers + training config |

---

### 4.11 Cell 11 — Live Demo

```python
def extract_features(target_model, tokenizer, review_text):
    # Builds the inference prompt (no response included)
    # Runs model.generate() with greedy decoding
    # Returns the text after "### Response:"
```

Five carefully chosen reviews are tested on both models:

| Review type | What it tests |
|---|---|
| Negative with pretty words | Sentiment on text that has positive vocabulary but negative meaning |
| Balanced-sounding but negative | Sentiment when the review starts positively then turns |
| Acting + plot + music all present | Whether all 3 aspects are detected |
| Very short review | Format compliance when input is minimal |
| ALL-CAPS + exclamation marks | Whether intensity is scored as "strong" correctly |

For each review the cell prints both model outputs and flags with `✓ Models disagree` when they differ — making it immediately obvious which reviews surface the fine-tuning improvement.

---

## 5. Results & Metrics

### Full Results Table

| Metric | Base Model | LoRA Fine-Tuned | Gain |
|---|---|---|---|
| Format compliance | 97.3% | **100.0%** | +2.7% |
| Sentiment accuracy | 70.7% | **80.7%** | **+10.0%** |
| Emotion hit rate | 69.3% | 69.3% | 0% |
| Aspect hit rate | 91.3% | **100.0%** | **+8.7%** |
| Intensity accuracy | 32.0% | **40.7%** | +8.7% |

### Sentiment Classification Report

**Base model:**
```
              precision  recall  f1-score  support
negative          0.79    0.65      0.71       77
positive          0.67    0.81      0.74       69
accuracy                            0.73      146
```

**Fine-tuned:**
```
              precision  recall  f1-score  support
negative          0.86    0.76      0.81       79
positive          0.76    0.86      0.81       71
accuracy                            0.81      150
```

### Interpretation

The largest gains are in **sentiment** (+10%) and **aspect detection** (+8.7%). Emotion hit rate did not improve because the base model already converges on a reasonable emotion set from its general instruction-following training, and 8,000 samples over 3 epochs is not enough to meaningfully shift emotion predictions on this 7-class problem. Intensity remains the hardest field for both models — it is inherently more subjective and the rule-based labels contain noise.

---

## 6. Design Decisions & Trade-offs

### Why Qwen2.5-0.5B?

It is the smallest model that reliably follows instruction prompts in English and produces structured output. Smaller models (e.g., GPT-2) cannot follow the output format at all. Larger models (e.g., Qwen2.5-1.5B) would push accuracy higher but double training time on a T4 and risk Kaggle session timeouts.

### Why Rule-Based Feature Labels?

A cleaner approach would be to use a stronger model (GPT-4, Claude) to annotate the IMDB dataset with emotions, aspects, and intensity. However, that requires API access and cost. Rule-based labeling is self-contained, reproducible, and free — appropriate for a demonstration project. The label quality is the main ceiling on intensity accuracy.

### Why Greedy Decoding at Inference?

For classification tasks, `do_sample=False` (greedy) gives deterministic, reproducible results. Temperature sampling adds randomness that can flip a correct answer to incorrect on any given run. Since we are measuring accuracy, consistency matters more than diversity.

### Why LoRA Alpha = 2 × Rank?

`lora_alpha` controls the scaling of the LoRA output: `(alpha/r) × B × A`. Setting `alpha = 2 × r` gives a fixed scale of 2.0 regardless of rank, meaning the effective learning rate of the LoRA update stays constant as we change `r`. This is a widely used convention that makes hyperparameter transfer between different rank values more predictable.

---

## 7. Limitations

**Label noise on intensity.** The 4-signal heuristic is better than the original 2-signal version but still imperfect. A review can be emotionally intense yet short and calm in tone ("Awful. Do not watch.") — the heuristic will call this "weak". This is the primary ceiling on intensity accuracy.

**Emotion labels are top-2 only.** A review can express more than 2 emotions. Capping at 2 means the ground truth may miss emotions that a model correctly predicts, artificially lowering the emotion hit rate metric.

**Test set size.** 150 samples is sufficient for directional comparison but margins of error are ±4–5%. Differences below 5 percentage points should not be over-interpreted.

**Domain specificity.** The model was fine-tuned on movie reviews. It will perform worse on product reviews, restaurant reviews, or other domains without additional fine-tuning.

**Emotion hit rate did not improve.** Three epochs on 8,000 samples was enough to teach the model the output format and improve sentiment and aspect detection, but not enough to substantially shift which emotions it associates with which reviews. More epochs or emotion-specific training examples would be needed.
