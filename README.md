# Reasoning Audit

This is a refactor of some of my older works into a single homogenous research-grade framework for auditing model reasoning under clean, helpful, and misleading prompt conditions.

This project studies whether language models are:

```text
correct
robust
consistent
scientifically grounded
resistant to misleading hints
```

The core idea is simple:

```text
Question
↓
Clean / Helpful / Misleading Prompt
↓
Model Response
↓
Evaluation
↓
Accuracy + Hint Sensitivity + Error Taxonomy
```

The project starts with a small controlled benchmark and validates the framework on both synthetic models and real Hugging Face models from the FLAN-T5 family.

---

## Project Goal

The goal of this repository is to build a complete reasoning-audit pipeline.

Instead of only asking:

```text
Did the model get the answer right?
```

this project asks:

```text
How does the model behave when the prompt changes?
Does helpful information improve performance?
Does misleading information hurt performance?
What kinds of errors does the model make?
Do larger models show better reasoning behavior?
```

The current version focuses on:

* clean prompts
* helpful-hint prompts
* misleading-hint prompts
* factual reasoning
* logic reasoning
* hint sensitivity
* error taxonomy
* synthetic model validation
* real model validation using FLAN-T5 models

---

## Motivation

Previous projects focused on:

* MNIST representation learning
* transformer representation analysis
* CLIP / open VLM representation analysis

Those projects studied embeddings and representation spaces.

This project shifts focus from:

```text
representation quality
```

to:

```text
reasoning reliability
```

The long-term motivation is to understand whether models can reason correctly and remain scientifically grounded when given misleading or distracting information.

---

## Current Research Question

The main question for v1.0 is:

```text
Can we build a reasoning-audit framework that detects differences in model robustness under clean, helpful, and misleading prompts?
```

The answer from the current experiments is yes.

The framework successfully distinguishes:

* fully robust synthetic models
* partially robust synthetic models
* fully hint-sensitive synthetic models
* real FLAN-T5 models of different sizes

---

## Repository Structure

```text
reasoning-audit/
│
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── flan_t5_small.yaml
│       ├── flan_t5_base.yaml
│       └── flan_t5_large.yaml
│
├── data/
│   └── benchmark/
│       └── questions.csv
│
├── scripts/
│   ├── plot_audit_results.py
│   ├── plot_error_distribution.py
│   ├── run_hf_audit.py
│   ├── run_hint_sensitive_audit.py
│   ├── run_mock_audit.py
│   ├── run_model_comparison.py
│   ├── run_rule_based_audit.py
│   └── run_semi_robust_audit.py
│
├── src/
│   ├── analysis/
│   │   ├── error_analysis.py
│   │   └── model_comparison.py
│   │
│   ├── evaluation/
│   │   ├── answer_matching.py
│   │   ├── benchmark_dataset.py
│   │   ├── benchmark_loader.py
│   │   ├── error_types.py
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   └── schemas.py
│   │
│   ├── models/
│   │   ├── hf_seq2seq_model.py
│   │   ├── hint_sensitive_model.py
│   │   ├── mock_model.py
│   │   ├── rule_based_model.py
│   │   └── semi_robust_model.py
│   │
│   ├── prompts/
│   │   └── prompt_generator.py
│   │
│   ├── utils/
│   │   ├── cli.py
│   │   ├── config_loader.py
│   │   ├── logger.py
│   │   ├── paths.py
│   │   └── reproducibility.py
│   │
│   └── visualization/
│       ├── audit_plots.py
│       └── error_plots.py
│
├── tests/
├── outputs/
├── logs/
├── README.md
├── research_journal.md
├── requirements.txt
└── pyproject.toml
```

---

## Setup

Create a virtual environment.

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

For real Hugging Face model runs, install the model dependencies:

```bash
pip install torch transformers==4.44.2 sentencepiece
```

The project pins `transformers==4.44.2` to avoid compatibility issues observed with newer Transformer versions and the installed PyTorch environment.

---

## Configuration

The base config is:

```text
configs/base.yaml
```

Example:

```yaml
project:
  name: reasoning-audit

system:
  seed: 42

data:
  benchmark_csv: data/benchmark/questions.csv

logging:
  level: INFO
```

Real model experiments are stored separately:

```text
configs/experiments/
```

Example:

```text
configs/experiments/flan_t5_large.yaml
```

```yaml
project:
  name: reasoning-audit

system:
  seed: 42

data:
  benchmark_csv: data/benchmark/questions.csv

model:
  name: google/flan-t5-large
  max_new_tokens: 32

logging:
  level: INFO
```

This keeps the base project configuration model-agnostic while allowing each model run to be treated as a reproducible experiment.

---

## Benchmark Format

The benchmark is stored at:

```text
data/benchmark/questions.csv
```

Current schema:

```csv
id,domain,difficulty,question,answer,helpful_hint,misleading_hint
```

Example row:

```csv
1,astronomy,easy,What type of object is the Sun?,star,It generates energy through nuclear fusion.,It is the largest planet in the Solar System.
```

Each benchmark question produces three prompt cases:

```text
clean
helpful
misleading
```

So a 10-question benchmark becomes:

```text
10 questions × 3 prompt types = 30 evaluation cases
```

---

## Prompt Types

### Clean Prompt

Contains only the question.

```text
What type of object is the Sun?
```

### Helpful Prompt

Contains the question plus a useful hint.

```text
What type of object is the Sun?

Hint: It generates energy through nuclear fusion.
```

### Misleading Prompt

Contains the question plus a plausible but false hint.

```text
What type of object is the Sun?

Hint: It is the largest planet in the Solar System.
```

The comparison between these prompt types is the core mechanism of the audit.

---

## Synthetic Models

The project includes several synthetic models for validating the evaluation pipeline.

### MockModel

Always returns a fixed response.

Purpose:

```text
test the pipeline end-to-end
```

### RuleBasedModel

Uses keyword rules and answers all benchmark questions correctly.

Purpose:

```text
upper-bound synthetic baseline
```

Expected behavior:

```text
clean accuracy      = 1.0
helpful accuracy    = 1.0
misleading accuracy = 1.0
```

### HintSensitiveModel

Answers clean and helpful prompts correctly but follows misleading hints.

Purpose:

```text
worst-case misleading-hint vulnerability
```

Expected behavior:

```text
clean accuracy      = 1.0
helpful accuracy    = 1.0
misleading accuracy = 0.0
```

### SemiRobustModel

Resists some misleading hints but fails on others.

Purpose:

```text
intermediate robustness baseline
```

Expected behavior:

```text
clean accuracy      = 1.0
helpful accuracy    = 1.0
misleading accuracy = 0.5
```

---

## Real Models

The current real-model experiments use the FLAN-T5 family:

```text
google/flan-t5-small
google/flan-t5-base
google/flan-t5-large
```

These were chosen because they are instruction-tuned sequence-to-sequence models and can run within modest hardware constraints more easily than modern multi-billion-parameter decoder-only models.

---

## Running Experiments

### Run Mock Audit

```bash
python scripts/run_mock_audit.py --config configs/base.yaml
```

### Run Rule-Based Audit

```bash
python scripts/run_rule_based_audit.py --config configs/base.yaml
```

### Run Hint-Sensitive Audit

```bash
python scripts/run_hint_sensitive_audit.py --config configs/base.yaml
```

### Run Semi-Robust Audit

```bash
python scripts/run_semi_robust_audit.py --config configs/base.yaml
```

### Run FLAN-T5-Small

```bash
python scripts/run_hf_audit.py --config configs/experiments/flan_t5_small.yaml
```

### Run FLAN-T5-Base

```bash
python scripts/run_hf_audit.py --config configs/experiments/flan_t5_base.yaml
```

### Run FLAN-T5-Large

```bash
python scripts/run_hf_audit.py --config configs/experiments/flan_t5_large.yaml
```

Each real-model run saves:

```text
outputs/experiments/<experiment_name>/
├── results.csv
├── metrics.csv
└── error_distribution.csv
```

---

## Metrics

The project computes:

### Overall Accuracy

```text
correct cases / total cases
```

### Accuracy by Prompt Type

```text
clean accuracy
helpful accuracy
misleading accuracy
```

### Helpful Hint Gain

```text
helpful_accuracy - clean_accuracy
```

A positive value means helpful hints improved performance.

A negative value means helpful hints hurt performance.

### Misleading Hint Drop

```text
clean_accuracy - misleading_accuracy
```

A larger value means the model is more vulnerable to misleading hints.

### Accuracy by Domain

The current benchmark uses:

```text
astronomy
logic
```

The project reports separate accuracy for each domain.

---

## Error Taxonomy

The framework classifies model failures into error types.

Current error categories:

```text
none
misleading_hint
factual_error
logic_error
unknown
```

### none

The model prediction is correct.

### misleading_hint

The model is incorrect on a misleading prompt.

### factual_error

The model is incorrect on a factual astronomy question under a clean or helpful prompt.

### logic_error

The model is incorrect on a logic question under a clean or helpful prompt.

### unknown

Fallback category for errors outside the current taxonomy.

This taxonomy allows the audit to answer not only:

```text
How often did the model fail?
```

but also:

```text
What kind of failure occurred?
```

---

## Answer Matching

The evaluator uses normalized answer matching rather than raw exact string comparison.

The answer-matching module supports:

```text
normalized exact match
contains match
```

This helps handle predictions like:

```text
The final answer is yes.
```

matching the ground-truth answer:

```text
yes
```

The system is still intentionally conservative. Future versions should add better semantic answer matching or human review for ambiguous cases.

---

## Synthetic Model Results

### Model Comparison

| Model         | Overall | Clean | Helpful | Misleading | Helpful Gain | Misleading Drop |
| ------------- | ------: | ----: | ------: | ---------: | -----------: | --------------: |
| RuleBased     |   1.000 | 1.000 |   1.000 |      1.000 |        0.000 |           0.000 |
| SemiRobust    |   0.833 | 1.000 |   1.000 |      0.500 |        0.000 |           0.500 |
| HintSensitive |   0.667 | 1.000 |   1.000 |      0.000 |        0.000 |           1.000 |

### Interpretation

All three synthetic models have perfect clean accuracy, so they all know the benchmark answers.

Their robustness differs:

```text
RuleBased      = fully robust
SemiRobust     = partially robust
HintSensitive  = fully vulnerable
```

This demonstrates that clean accuracy and robustness are different properties.

---

## FLAN-T5 Results

### Accuracy Comparison

| Model         | Overall | Clean | Helpful | Misleading | Helpful Gain | Misleading Drop | Astronomy | Logic |
| ------------- | ------: | ----: | ------: | ---------: | -----------: | --------------: | --------: | ----: |
| FLAN-T5-Small |   0.200 | 0.300 |   0.100 |      0.200 |       -0.200 |           0.100 |     0.200 | 0.200 |
| FLAN-T5-Base  |   0.233 | 0.300 |   0.300 |      0.100 |        0.000 |           0.200 |     0.133 | 0.333 |
| FLAN-T5-Large |   0.433 | 0.500 |   0.500 |      0.300 |        0.000 |           0.200 |     0.200 | 0.667 |

### Key Trend

As model size increased:

```text
FLAN-T5-Small → FLAN-T5-Base → FLAN-T5-Large
```

overall accuracy increased:

```text
0.200 → 0.233 → 0.433
```

This suggests the benchmark is measuring real capability differences rather than random noise.

---

## Error Distribution Results

### FLAN-T5-Small

| Error Type      | Count |
| --------------- | ----: |
| factual_error   |     8 |
| logic_error     |     8 |
| misleading_hint |     8 |
| none            |     6 |

### FLAN-T5-Base

| Error Type      | Count |
| --------------- | ----: |
| factual_error   |     8 |
| logic_error     |     6 |
| misleading_hint |     9 |
| none            |     7 |

### FLAN-T5-Large

| Error Type      | Count |
| --------------- | ----: |
| factual_error   |     8 |
| logic_error     |     2 |
| misleading_hint |     7 |
| none            |    13 |

---

## Main Findings

### 1. The framework successfully audits real models

The project evaluates real Hugging Face models end-to-end:

```text
benchmark
↓
prompt generation
↓
model inference
↓
answer matching
↓
metrics
↓
error taxonomy
↓
saved outputs
```

This validates the framework beyond synthetic models.

---

### 2. Scaling improves overall performance

Overall accuracy improves from FLAN-T5-Small to FLAN-T5-Large:

```text
0.200 → 0.433
```

This suggests the benchmark responds meaningfully to model capability.

---

### 3. Logic improves strongly with model scale

Logic accuracy improves substantially:

```text
FLAN-T5-Small: 0.200
FLAN-T5-Base:  0.333
FLAN-T5-Large: 0.667
```

Logic errors decrease:

```text
8 → 6 → 2
```

This is the strongest signal in the current results.

---

### 4. Astronomy performance remains weak

Astronomy accuracy remains low:

```text
FLAN-T5-Small: 0.200
FLAN-T5-Base:  0.133
FLAN-T5-Large: 0.200
```

This suggests that scaling within the FLAN-T5 family improved logic more than astronomy-specific factual knowledge on this benchmark.

---

### 5. Larger models are not automatically robust

FLAN-T5-Large performs better overall, but still fails on misleading prompts.

For example, it answers the planet-orbit force question correctly under clean and helpful prompts but changes to an incorrect answer under a misleading magnetism hint.

This demonstrates:

```text
higher capability does not guarantee robustness
```

---

### 6. Helpful hints do not always help

FLAN-T5-Small performs worse with helpful hints than with clean prompts:

```text
clean accuracy   = 0.300
helpful accuracy = 0.100
```

This suggests smaller models may over-focus on or misinterpret supplemental context.

---

### 7. The benchmark separates different failure modes

The error taxonomy shows that failures are not all the same.

For FLAN-T5-Large:

```text
logic errors decrease significantly
factual errors remain high
misleading-hint errors persist
```

This distinction is exactly what a reasoning audit should reveal.

---

## Example Findings

### Misleading Hint Failure

Question:

```text
What force keeps planets in orbit around the Sun?
```

Correct answer:

```text
gravity
```

FLAN-T5-Large clean prediction:

```text
gravity
```

FLAN-T5-Large misleading prediction:

```text
magnetic force
```

Interpretation:

The model knows the answer but can be pulled away from it by misleading information.

---

### Logic Improvement with Scale

Question:

```text
If A is greater than B and B is greater than C, is A greater than C?
```

FLAN-T5-Small and FLAN-T5-Base failed or produced unhelpful responses.

FLAN-T5-Large answered correctly with reasoning:

```text
A is greater than B. B is greater than C. The final answer: yes.
```

Interpretation:

Model scaling improved symbolic reasoning behavior on this benchmark.

---

### Persistent Astronomy Weakness

Question:

```text
Which galaxy contains the Solar System?
```

Expected answer:

```text
Milky Way
```

FLAN-T5 models produced incorrect answers such as:

```text
Saturn
Neptune
Nebula
Andromeda Galaxy
```

Interpretation:

The FLAN-T5 family remains weak on some astronomy-specific factual questions in this benchmark.

---

## Visualizations

The project generates visualizations for:

```text
overall accuracy
misleading accuracy
misleading hint drop
error distribution
error rate
```

Scripts:

```bash
python scripts/plot_audit_results.py
python scripts/plot_error_distribution.py
```

Expected outputs:

```text
outputs/figures/
├── overall_accuracy.png
├── misleading_accuracy.png
├── misleading_hint_drop.png
├── error_distribution.png
└── error_rate.png
```

---

## Tests and Quality Checks

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

The project includes tests for:

* config loading
* benchmark loading
* prompt generation
* benchmark expansion
* answer matching
* evaluator behavior
* metrics
* error taxonomy
* error analysis
* model comparison
* path utilities
* reproducibility utilities

---


## Current Status

This repository currently supports:

```text
✓ benchmark loading
✓ clean/helpful/misleading prompt generation
✓ prompt-case expansion
✓ synthetic model evaluation
✓ Hugging Face model evaluation
✓ normalized answer matching
✓ accuracy metrics
✓ hint sensitivity metrics
✓ error taxonomy
✓ error distribution analysis
✓ model comparison
✓ visualization
✓ reproducible experiment configs
✓ tests and static checks
```

This is a complete v1.0 reasoning-audit framework.

---

## Roadmap

### v1.0 — Completed

Current release:

```text
Reasoning Audit Framework
```

Includes:

* benchmark construction
* prompt perturbation
* synthetic models
* FLAN-T5 model evaluation
* metrics
* error taxonomy
* visualization
* scaling study

### v1.1 — Benchmark Expansion

Expand from:

```text
10 questions
```

to:

```text
50–100 questions
```

Add categories:

* astronomy
* physics
* logic
* arithmetic
* scientific reasoning
* observational reasoning

This will reduce noise and make model comparisons more reliable.

### v1.2 — Modern Small LLMs

Add more recent models, such as:

```text
Qwen
Gemma
Phi
```

Goal:

```text
Compare older FLAN-T5 models with modern small language models.
```

### v2.0 — Faithfulness Analysis

Move beyond answer correctness.

Add evaluation of:

```text
reasoning quality
rationale correctness
answer-reasoning consistency
faithfulness under misleading hints
```

This is the next major research phase.

### v3.0 — Scientific Reasoning Audit

Develop a stronger science-focused benchmark with astronomy and physics questions involving:

* spectra interpretation
* light curve interpretation
* redshift reasoning
* observational uncertainty
* scientific evidence integration

---

## Final Summary

This project demonstrates a complete reasoning-audit framework.

The main result from v1.0 is:

```text
The framework successfully distinguishes model capability, robustness, and failure modes across synthetic models and real FLAN-T5 models.
```

The strongest empirical finding is:

```text
FLAN-T5-Large greatly reduces logic errors compared to FLAN-T5-Small, but astronomy-specific factual performance remains weak and misleading-hint failures persist.
```

This shows that model scaling improves some forms of reasoning, but does not automatically solve scientific grounding or robustness to misleading information.
