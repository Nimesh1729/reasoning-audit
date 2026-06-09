# Reasoning Audit

This is a research-grade framework for evaluating language-model reasoning under clean, helpful, and misleading prompt conditions.

This project studies whether models are not only correct, but also robust to prompt perturbations. It supports free-form benchmarks, multiple-choice benchmarks, Hugging Face models, causal language models, LoRA fine-tuning, train/test evaluation, and error analysis.

---

## Project Summary

The core idea is simple:

```text
Same question
    ↓
Clean prompt
Helpful-hint prompt
Misleading-hint prompt
    ↓
Model response
    ↓
Accuracy + robustness analysis
```

The project measures:

```text
overall accuracy
clean accuracy
helpful accuracy
misleading accuracy
helpful hint gain
misleading hint drop
domain accuracy
error type distribution
```

The final benchmark uses multiple-choice questions with objective option-letter evaluation.

---

## Motivation

Standard accuracy answers:

```text
Did the model get the answer right?
```

This project asks more:

```text
Does helpful information improve the answer?
Does misleading information hurt the answer?
Does fine-tuning improve held-out performance?
Do models fail differently across domains?
Is the evaluator itself reliable?
```

The project builds on earlier work in representation learning and model analysis, but it is valuable as a standalone reasoning-audit framework.

---

## Repository Structure

```text
reasoning-audit/
│
├── configs/
│   ├── base.yaml
│   └── experiments/
│
├── data/
│   └── benchmark/
│       ├── questions.csv
│       ├── questions_v1.csv
│       ├── questions_v2.csv
│       ├── questions_mcq_pilot.csv
│       └── questions_v2_mcq.csv
│
├── scripts/
│   ├── run_hf_audit.py
│   ├── run_causal_lm_audit.py
│   ├── run_lora_audit.py
│   ├── train_flan_t5_lora.py
│   ├── split_benchmark.py
│   └── ...
│
├── src/
│   ├── analysis/
│   ├── evaluation/
│   ├── models/
│   ├── prompts/
│   ├── training/
│   ├── utils/
│   └── visualization/
│
├── tests/
├── outputs/
├── logs/
├── README.md
├── research_journal.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## Installation

It is recommended to create and activate a virtual environment.

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Install the project:

```bash
pip install -e ".[dev]"
```

For Hugging Face models and LoRA training:

```bash
pip install torch transformers==4.44.2 datasets accelerate peft==0.12.0 sentencepiece
```

The tested stable setup used:

```text
transformers==4.44.2
peft==0.12.0
```

---

## Development Commands

Run tests:

```bash
pytest
```

Run Ruff:

```bash
ruff check .
```

Run mypy:

```bash
mypy src
```

Recommended before committing:

```bash
pytest
ruff check .
mypy src
```

---

## Benchmark Design

The project originally used free-form answers, but free-form scoring introduced ambiguity.

Examples:

```text
0.25 vs 1/4
constant velocity vs velocity becomes constant
changing Sun-Earth-Moon geometry vs relative positions of the Sun, Earth, and Moon
```

To make evaluation more reliable, the final benchmark was redesigned as multiple choice.

Final MCQ format:

```text
Question:
What causes the phases of the Moon?

A. Earth's shadow causing every phase
B. Changing Sun-Earth-Moon geometry
C. The Moon producing its own light
D. Cloud cover on Earth

Answer key:
B
```

The model is instructed to answer only with:

```text
A, B, C, or D
```

This makes evaluation objective:

```text
prediction == answer_key
```

---

## Final Benchmark

Final benchmark file:

```text
data/benchmark/questions_v2_mcq.csv
```

It contains:

```text
100 questions
25 astronomy
25 logic
25 physics
25 arithmetic
```

Full benchmark answer-key balance:

```text
A: 25
B: 25
C: 25
D: 25
```

The train/test split is:

```text
80 training questions
20 test questions
```

The test set is balanced:

```text
5 questions per domain
5 answers per option letter
```

Generate the split with:

```bash
python scripts/split_benchmark.py
```

This creates:

```text
data/benchmark/train.csv
data/benchmark/test.csv
```

These are generated files and do not need to be tracked.

---

## Prompt Conditions

Each question is evaluated under three conditions.

### Clean

The question and options only.

### Helpful

The question, options, and a useful hint.

### Misleading

The question, options, and an incorrect or distracting hint.

This allows the benchmark to measure both accuracy and context sensitivity.

---

## Metrics

The main metrics are:

```text
overall_accuracy
clean_accuracy
helpful_accuracy
misleading_accuracy
helpful_hint_gain
misleading_hint_drop
domain_accuracy
```

Definitions:

```text
helpful_hint_gain = helpful_accuracy - clean_accuracy
misleading_hint_drop = clean_accuracy - misleading_accuracy
```

Interpretation:

```text
positive helpful_hint_gain → helpful hints improved performance
larger misleading_hint_drop → model was more vulnerable to misleading hints
```

---

## Error Taxonomy

Errors are classified as:

```text
none
misleading_hint
astronomy_error
logic_error
physics_error
arithmetic_error
unknown
```

This separates general correctness from the kind of failure made by the model.

---

## Models Evaluated

Synthetic validation models:

```text
MockModel
RuleBasedModel
HintSensitiveModel
SemiRobustModel
```

Real models:

```text
google/flan-t5-base
google/flan-t5-base + LoRA
Qwen/Qwen2.5-0.5B-Instruct
```

Earlier exploratory experiments also used:

```text
google/flan-t5-small
google/flan-t5-large
Qwen/Qwen2.5-1.5B-Instruct
```

Qwen 1.5B was not used for final results because it was too memory-heavy for comfortable local evaluation.

---

## Running Experiments

### FLAN-T5-Base MCQ Evaluation

```bash
python scripts/run_hf_audit.py --config configs/experiments/flan_t5_base_mcq_test.yaml
```

### Qwen2.5-0.5B MCQ Evaluation

```bash
python scripts/run_causal_lm_audit.py --config configs/experiments/qwen_0_5b_mcq_test.yaml
```

### Train FLAN-T5-Base LoRA

```bash
python scripts/train_flan_t5_lora.py --config configs/experiments/flan_t5_base_lora.yaml
```

### Evaluate FLAN-T5-Base + LoRA

```bash
python scripts/run_lora_audit.py --config configs/experiments/flan_t5_base_lora_mcq_eval.yaml
```

The final MCQ LoRA adapter is saved to:

```text
outputs/lora/flan_t5_base_mcq/
```

LoRA adapters are generated artifacts and are not intended to be committed.

---

## Final MCQ Results

Final results on the 20-question MCQ test split:

| Model                   | Overall | Clean | Helpful | Misleading | Helpful Gain | Misleading Drop |
| ----------------------- | ------: | ----: | ------: | ---------: | -----------: | --------------: |
| FLAN-T5-Base            |   0.300 | 0.300 |   0.400 |      0.200 |       +0.100 |           0.100 |
| FLAN-T5-Base + MCQ LoRA |   0.317 | 0.350 |   0.400 |      0.200 |       +0.050 |           0.150 |
| Qwen2.5-0.5B-Instruct   |   0.350 | 0.450 |   0.350 |      0.250 |       -0.100 |           0.200 |

Final ranking:

```text
1. Qwen2.5-0.5B-Instruct     0.350
2. FLAN-T5-Base + MCQ LoRA   0.317
3. FLAN-T5-Base              0.300
```

---

## Domain Results

| Model                   | Astronomy | Logic |      Physics |   Arithmetic |
| ----------------------- | --------: | ----: | -----------: | -----------: |
| FLAN-T5-Base            |     0.133 | 0.600 | not reported | not reported |
| FLAN-T5-Base + MCQ LoRA |     0.200 | 0.600 |        0.267 |        0.200 |
| Qwen2.5-0.5B-Instruct   |     0.667 | 0.267 |        0.267 |        0.200 |

Note: The FLAN baseline script reported astronomy and logic accuracy only. The LoRA and Qwen scripts reported all four domains.

---

## Main Findings

### 1. MCQ evaluation was more reliable than free-form evaluation

Free-form answer matching introduced false positives and false negatives. MCQ option-letter evaluation made scoring objective and auditable.

### 2. Qwen2.5-0.5B was the strongest final model

Qwen achieved the highest overall score on the final MCQ benchmark.

### 3. MCQ LoRA slightly improved FLAN-T5-Base

LoRA improved overall accuracy:

```text
0.300 → 0.317
```

and clean accuracy:

```text
0.300 → 0.350
```

### 4. LoRA did not improve misleading-hint robustness

Misleading accuracy stayed unchanged:

```text
0.200 → 0.200
```

This suggests that task adaptation and robustness are different.

### 5. Helpful hints did not always help

Qwen performed better on clean prompts than helpful prompts:

```text
clean:   0.450
helpful: 0.350
```

Extra context can sometimes distract small instruction models.

---

## Limitations

The final test set is small:

```text
20 questions
60 prompt cases
```

The MCQ format improves scoring reliability but can introduce answer-choice bias. This was mitigated by balancing answer keys.

LoRA training used only:

```text
80 training questions
```

so the fine-tuning result should be interpreted as a small-scale adaptation experiment.

The final model set is limited to a small number of models. Larger evaluations would require more models, more benchmark questions, and repeated splits.

---

## Reproducibility Notes

Tracked benchmark assets:

```text
data/benchmark/questions.csv
data/benchmark/questions_v1.csv
data/benchmark/questions_v2.csv
data/benchmark/questions_mcq_pilot.csv
data/benchmark/questions_v2_mcq.csv
```

Generated artifacts:

```text
data/benchmark/train.csv
data/benchmark/test.csv
outputs/
logs/
```

Recommended `.gitignore` entries:

```gitignore
data/benchmark/train.csv
data/benchmark/test.csv

outputs/
logs/

venv/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
```

---

## Version Timeline

### v1.0

Initial reasoning-audit framework:

```text
free-form benchmark
prompt variants
metrics
error taxonomy
synthetic models
FLAN experiments
```

### v2.0

Fine-tuning infrastructure:

```text
train/test split
LoRA training pipeline
held-out evaluation
before/after adaptation study
```

### v3.0

Final MCQ benchmark release:

```text
objective option-letter evaluation
100-question MCQ benchmark
balanced train/test split
MCQ LoRA experiment
Qwen comparison
final model comparison
```

---

## Final Conclusion

This project shows that reasoning evaluation requires more than standard accuracy.

The central methodological result is:

```text
Free-form answer matching was too ambiguous.
MCQ option-letter evaluation produced a more reliable benchmark.
```

The central empirical result is:

```text
Qwen2.5-0.5B-Instruct performed best on the final MCQ benchmark.
```

The central robustness result is:

```text
MCQ LoRA slightly improved FLAN-T5-Base accuracy, but did not improve robustness to misleading hints.
```

Overall, the repository provides a complete standalone framework for auditing language-model behavior under clean, helpful, and misleading context.
