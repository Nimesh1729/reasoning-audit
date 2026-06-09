# Research Journal — Reasoning Audit

## Project Overview

This project builds a research-grade framework for auditing model reasoning under different prompt conditions.

The central goal is to evaluate whether a model is:

```text
correct
robust
consistent
sensitive to hints
vulnerable to misleading information
scientifically grounded
```

Unlike previous representation-learning projects, this repository does not focus on embeddings. Instead, it focuses on model behavior under controlled prompt perturbations.

The basic audit structure is:

```text
Benchmark Question
        ↓
Clean / Helpful / Misleading Prompt
        ↓
Model Response
        ↓
Evaluation
        ↓
Accuracy + Hint Sensitivity + Error Taxonomy
```

---

## Motivation

Previous projects in the broader roadmap focused on representation learning:

```text
MNIST Representation Analysis
Transformer Representation Analysis
Open VLM / CLIP Representation Analysis
```

Those projects asked questions like:

```text
Do embeddings cluster?
Do related concepts separate?
How do representations change across layers?
```

This project asks a different type of question:

```text
Can a model reason correctly and remain robust when the prompt changes?
```

The motivation comes from reasoning-audit and scientific-grounding concerns:

* A model may answer correctly for the wrong reason.
* A model may be correct under clean prompts but fail under misleading hints.
* A model may over-trust false context.
* A model may appear capable but lack robustness.
* Larger models may improve reasoning while still failing scientific grounding.

This repository builds the first version of a framework for studying these behaviors.

---

## Core Research Question

The main question for v1.0 is:

```text
Can we build a reasoning-audit framework that detects differences in model robustness under clean, helpful, and misleading prompts?
```

Subquestions:

1. Can the framework generate multiple prompt variants from one benchmark question?
2. Can it evaluate model answers automatically?
3. Can it measure helpful-hint gain and misleading-hint drop?
4. Can it distinguish different failure modes?
5. Can it compare synthetic models with known behaviors?
6. Can it audit real Hugging Face models?
7. Does model scaling produce meaningful differences?

---

## Repository Design Philosophy

The project follows the same engineering standards used in earlier research repos:

* Google Python Style Guide
* type hints (not rigid)
* Ruff linting (extremely lax)
* mypy static checking (extremely lax)
* pytest tests
* config-driven experiments
* reproducible experiment structure
* centralized utilities
* clear separation between loading, evaluation, analysis, visualization, and models

The intended architecture is:

```text
configs/
data/
scripts/
src/
tests/
outputs/
logs/
README.md
research_journal.md
requirements.txt
pyproject.toml
```

This mirrors the structure used in the Transformer Representation and Open VLM projects, but adapted for reasoning evaluation rather than embedding analysis.

---

## Phase 0 — Infrastructure Setup

The project began with the standard research-repo infrastructure.

Implemented:

```text
pyproject.toml
requirements.txt
configs/base.yaml
src/utils/config_loader.py
src/utils/logger.py
src/utils/paths.py
src/utils/reproducibility.py
src/utils/cli.py
tests/
```

The project was installed in editable mode using:

```bash
pip install -e ".[dev]"
```

Development tools included:

```text
pytest
ruff
mypy
types-PyYAML
```

This provided the same reliable development foundation used in previous repos.

---

## Phase 1 — Benchmark Loader

The first project-specific component was the benchmark loader.

File:

```text
src/evaluation/benchmark_loader.py
```

Purpose:

```text
Load questions.csv
Validate required columns
Return benchmark DataFrame
```

Initial schema:

```csv
id,domain,difficulty,question,answer
```

Later expanded to:

```csv
id,domain,difficulty,question,answer,helpful_hint,misleading_hint
```

The final required columns are:

```text
id
domain
difficulty
question
answer
helpful_hint
misleading_hint
```

Tests were added to verify:

* valid benchmark loading
* error handling for missing columns

This established the benchmark as the foundation of the audit.

---

## Phase 2 — Prompt Generator

The next component was prompt generation.

File:

```text
src/prompts/prompt_generator.py
```

The prompt generator creates three prompt variants from each benchmark question:

```text
clean
helpful
misleading
```

Example question:

```text
What type of object is the Sun?
```

Clean prompt:

```text
What type of object is the Sun?
```

Helpful prompt:

```text
What type of object is the Sun?

Hint: It generates energy through nuclear fusion.
```

Misleading prompt:

```text
What type of object is the Sun?

Hint: It is the largest planet in the Solar System.
```

This prompt-variant design is the core audit mechanism.

Tests were added for:

* clean prompt generation
* helpful prompt generation
* misleading prompt generation

---

## Phase 3 — Benchmark Dataset Expansion

The benchmark dataset module expands benchmark rows into prompt cases.

File:

```text
src/evaluation/benchmark_dataset.py
```

Function:

```text
build_prompt_cases()
```

For each benchmark row, it creates:

```text
1 clean case
1 helpful case
1 misleading case
```

Therefore:

```text
10 benchmark questions × 3 prompt types = 30 prompt cases
```

This converted the benchmark from a static question list into an evaluation dataset.

Each prompt case contains:

```text
question_id
domain
difficulty
answer
prompt_type
prompt
```

Tests verified that one benchmark row expands into exactly three prompt cases.

---

## Phase 4 — Evaluation Schema

A structured evaluation result was introduced.

File:

```text
src/evaluation/schemas.py
```

Schema:

```python
@dataclass
class EvaluationResult:
    question_id: int
    domain: str
    prompt_type: str
    ground_truth: str
    prediction: str
    correct: bool
    error_type: ErrorType = ErrorType.NONE
```

A dataclass was used because each result is a structured data record.

The schema represents one model prediction on one prompt case.

This became the central object passed into metrics, error analysis, and comparison functions.

---

## Phase 5 — Metrics

Evaluation metrics were implemented in:

```text
src/evaluation/metrics.py
```

Implemented metrics:

```text
compute_accuracy()
compute_accuracy_by_prompt_type()
compute_accuracy_by_domain()
compute_helpful_hint_gain()
compute_misleading_hint_drop()
```

Definitions:

```text
overall_accuracy = correct / total
```

```text
helpful_hint_gain = helpful_accuracy - clean_accuracy
```

```text
misleading_hint_drop = clean_accuracy - misleading_accuracy
```

Interpretation:

* Positive helpful hint gain means helpful hints improved performance.
* Negative helpful hint gain means helpful hints hurt performance.
* Higher misleading hint drop means stronger vulnerability to misleading information.

Tests were added for all metric functions.

---

## Phase 6 — Evaluator

The evaluator connects prompt cases to model predictions.

File:

```text
src/evaluation/evaluator.py
```

Pipeline:

```text
prompt case
        ↓
model.generate(prompt)
        ↓
prediction
        ↓
answer matching
        ↓
correct / incorrect
        ↓
error classification
        ↓
EvaluationResult
```

Initially, evaluation used strict normalized exact match.

Later, answer matching was moved into a dedicated module.

---

## Phase 7 — Answer Matching

A weakness appeared during real-model evaluation: exact string matching was too strict.

Example:

```text
Ground truth: yes
Prediction: The Sun is luminous.
```

The prediction may be semantically correct but fail exact matching.

To improve this, answer matching was moved into:

```text
src/evaluation/answer_matching.py
```

Implemented functions:

```text
normalize_answer()
exact_match()
contains_match()
answer_matches()
```

The evaluator now uses:

```text
answer_matches(prediction, ground_truth)
```

rather than raw string equality.

Current matching supports:

* normalized exact match
* contains match

This remains conservative and should be improved in future versions using semantic matching or human review.

---

## Phase 8 — Synthetic Models

Before integrating real models, synthetic models were created to validate the framework.

### MockModel

File:

```text
src/models/mock_model.py
```

Behavior:

```text
Always returns a fixed response.
```

Purpose:

```text
Verify end-to-end pipeline execution.
```

When set to always return `"star"`, only the Sun question was correct across its prompt variants.

Expected result:

```text
3 / 30 correct = 0.10 accuracy
```

This verified that benchmark loading, prompt generation, model generation, evaluation, and metrics worked together.

---

### RuleBasedModel

File:

```text
src/models/rule_based_model.py
```

Behavior:

```text
Uses keyword rules to answer all benchmark questions correctly.
```

Expected behavior:

```text
clean accuracy      = 1.0
helpful accuracy    = 1.0
misleading accuracy = 1.0
```

Purpose:

```text
Synthetic upper-bound baseline.
```

This model ignores misleading hints and always answers from the question.

---

### HintSensitiveModel

File:

```text
src/models/hint_sensitive_model.py
```

Behavior:

```text
Answers clean and helpful prompts correctly,
but follows misleading hints when present.
```

Observed result:

```text
overall_accuracy       = 0.6667
clean_accuracy         = 1.0
helpful_accuracy       = 1.0
misleading_accuracy    = 0.0
helpful_hint_gain      = 0.0
misleading_hint_drop   = 1.0
```

Interpretation:

The model is capable under clean and helpful prompts but fully vulnerable to misleading hints.

This produced the first true audit result:

```text
Capability does not imply robustness.
```

---

### SemiRobustModel

File:

```text
src/models/semi_robust_model.py
```

Behavior:

```text
Resists some misleading hints but fails on others.
```

Observed result:

```text
overall_accuracy       = 0.8333
clean_accuracy         = 1.0
helpful_accuracy       = 1.0
misleading_accuracy    = 0.5
helpful_hint_gain      = 0.0
misleading_hint_drop   = 0.5
```

Interpretation:

This model sits between RuleBasedModel and HintSensitiveModel.

It provides an intermediate robustness baseline.

---

## Phase 9 — Synthetic Model Comparison

Model comparison was implemented in:

```text
src/analysis/model_comparison.py
scripts/run_model_comparison.py
```

Synthetic comparison results:

| Model         | Overall | Clean | Helpful | Misleading | Helpful Gain | Misleading Drop |
| ------------- | ------: | ----: | ------: | ---------: | -----------: | --------------: |
| RuleBased     |   1.000 | 1.000 |   1.000 |      1.000 |        0.000 |           0.000 |
| SemiRobust    |   0.833 | 1.000 |   1.000 |      0.500 |        0.000 |           0.500 |
| HintSensitive |   0.667 | 1.000 |   1.000 |      0.000 |        0.000 |           1.000 |

Main interpretation:

```text
All synthetic models have perfect clean accuracy,
but they differ dramatically in misleading-prompt robustness.
```

This validates the purpose of the audit:

```text
clean accuracy and robustness are different properties.
```

---

## Phase 10 — Error Taxonomy

The next step was to move beyond correct/incorrect evaluation.

File:

```text
src/evaluation/error_types.py
```

Error categories:

```text
none
misleading_hint
factual_error
logic_error
unknown
```

Classification rule:

```text
correct prediction → none
wrong misleading prompt → misleading_hint
wrong astronomy clean/helpful prompt → factual_error
wrong logic clean/helpful prompt → logic_error
fallback → unknown
```

This introduced failure-mode analysis.

Now the framework can answer:

```text
What kind of failure occurred?
```

not only:

```text
Did the model fail?
```

---

## Phase 11 — Error Analysis

Error distribution analysis was implemented in:

```text
src/analysis/error_analysis.py
```

Function:

```text
compute_error_distribution()
```

Synthetic error distributions:

### HintSensitiveModel

```csv
error_type,count
none,20
misleading_hint,10
```

### SemiRobustModel

```csv
error_type,count
none,25
misleading_hint,5
```

Interpretation:

The error taxonomy correctly captured that all failures in these models were misleading-hint failures.

---

## Phase 12 — Visualizations

Visualization utilities were added in:

```text
src/visualization/audit_plots.py
src/visualization/error_plots.py
```

Scripts:

```text
scripts/plot_audit_results.py
scripts/plot_error_distribution.py
```

Generated figures:

```text
overall_accuracy.png
misleading_accuracy.png
misleading_hint_drop.png
error_distribution.png
error_rate.png
```

The plots visually showed the robustness ordering:

```text
RuleBased > SemiRobust > HintSensitive
```

The error-rate plot showed:

```text
HintSensitive: 33.3% misleading errors
SemiRobust:    16.7% misleading errors
```

Visualization completed the benchmark → evaluation → analysis → plot pipeline.

---

## Phase 13 — Hugging Face Model Integration

After synthetic validation, a real model wrapper was added.

File:

```text
src/models/hf_seq2seq_model.py
```

Model class:

```text
HFSeq2SeqModel
```

Supported Hugging Face seq2seq models using:

```text
AutoTokenizer
AutoModelForSeq2SeqLM
```

Prompt formatting:

```text
Answer with only the final answer.

{prompt}
```

The model wrapper supports:

```text
model_name
max_new_tokens
cuda/cpu device selection
```

Real model experiments were placed in:

```text
configs/experiments/
```

This kept `base.yaml` model-agnostic and treated each real model as an experiment.

---

## Phase 14 — FLAN-T5-Small Experiment

Experiment config:

```text
configs/experiments/flan_t5_small.yaml
```

Model:

```text
google/flan-t5-small
```

Results:

```csv
overall_accuracy,clean_accuracy,helpful_accuracy,misleading_accuracy,helpful_hint_gain,misleading_hint_drop,astronomy_accuracy,logic_accuracy
0.2,0.3,0.1,0.2,-0.2,0.1,0.2,0.2
```

Error distribution:

```csv
error_type,count
factual_error,8
misleading_hint,8
none,6
logic_error,8
```

Interpretation:

FLAN-T5-Small struggled broadly.

Key findings:

* Overall accuracy was only 20%.
* Helpful hints hurt performance.
* Misleading hints slightly reduced performance.
* Astronomy and logic accuracy were both 20%.
* Errors were evenly distributed across factual, logic, and misleading-hint failures.

This was the first real-model validation of the audit framework.

---

## Phase 15 — FLAN-T5-Base Experiment

Experiment config:

```text
configs/experiments/flan_t5_base.yaml
```

Model:

```text
google/flan-t5-base
```

Results:

```csv
overall_accuracy,clean_accuracy,helpful_accuracy,misleading_accuracy,helpful_hint_gain,misleading_hint_drop,astronomy_accuracy,logic_accuracy
0.2333,0.3,0.3,0.1,0.0,0.2,0.1333,0.3333
```

Error distribution:

```csv
error_type,count
factual_error,8
misleading_hint,9
none,7
logic_error,6
```

Interpretation:

FLAN-T5-Base improved slightly over Small.

Important observations:

* Overall accuracy improved from 0.20 to 0.233.
* Helpful accuracy improved from 0.10 to 0.30.
* Logic accuracy improved from 0.20 to 0.333.
* Logic errors decreased from 8 to 6.
* Misleading-hint failures increased from 8 to 9.
* Astronomy accuracy remained weak.

This suggested that larger FLAN-T5 capacity improved some reasoning behavior but did not necessarily improve robustness.

---

## Phase 16 — FLAN-T5-Large Experiment

Experiment config:

```text
configs/experiments/flan_t5_large.yaml
```

Model:

```text
google/flan-t5-large
```

Results:

```csv
overall_accuracy,clean_accuracy,helpful_accuracy,misleading_accuracy,helpful_hint_gain,misleading_hint_drop,astronomy_accuracy,logic_accuracy
0.4333,0.5,0.5,0.3,0.0,0.2,0.2,0.6667
```

Error distribution:

```csv
error_type,count
factual_error,8
none,13
misleading_hint,7
logic_error,2
```

Interpretation:

FLAN-T5-Large produced the strongest real-model result.

Key findings:

* Overall accuracy increased to 43.3%.
* Clean accuracy increased to 50%.
* Helpful accuracy increased to 50%.
* Misleading accuracy increased to 30%.
* Logic accuracy increased to 66.7%.
* Logic errors dropped sharply from 8 to 2 compared to Small.
* Astronomy accuracy remained at 20%.
* Factual errors remained high.
* Misleading-hint failures persisted.

This was the strongest validation that the benchmark responds to model capability.

---

## Phase 17 — FLAN-T5 Scaling Study

Real-model comparison:

| Model         | Overall | Clean | Helpful | Misleading | Helpful Gain | Misleading Drop | Astronomy | Logic |
| ------------- | ------: | ----: | ------: | ---------: | -----------: | --------------: | --------: | ----: |
| FLAN-T5-Small |   0.200 | 0.300 |   0.100 |      0.200 |       -0.200 |           0.100 |     0.200 | 0.200 |
| FLAN-T5-Base  |   0.233 | 0.300 |   0.300 |      0.100 |        0.000 |           0.200 |     0.133 | 0.333 |
| FLAN-T5-Large |   0.433 | 0.500 |   0.500 |      0.300 |        0.000 |           0.200 |     0.200 | 0.667 |

Scaling trend:

```text
FLAN-T5-Small → FLAN-T5-Base → FLAN-T5-Large
0.200         → 0.233        → 0.433
```

Main finding:

```text
Larger FLAN-T5 models performed better overall,
especially on logic reasoning.
```

Logic accuracy trend:

```text
0.200 → 0.333 → 0.667
```

Logic error trend:

```text
8 → 6 → 2
```

This is the strongest empirical signal in the project.

---

## Key Experimental Findings

### Finding 1 — The Framework Works End-to-End

The project successfully evaluates:

```text
synthetic models
real Hugging Face models
```

through the same pipeline:

```text
benchmark
prompt generation
model inference
answer matching
metrics
error taxonomy
visualization
```

This validates the framework design.

---

### Finding 2 — Capability and Robustness Are Different

Synthetic models showed that models can have identical clean accuracy but different robustness.

Example:

```text
RuleBased:      clean = 1.0, misleading = 1.0
HintSensitive:  clean = 1.0, misleading = 0.0
```

This demonstrates that clean accuracy alone is insufficient for reasoning audits.

---

### Finding 3 — FLAN-T5 Models Show Meaningful Scaling

Overall accuracy improved with model size:

```text
Small: 0.200
Base:  0.233
Large: 0.433
```

This suggests the benchmark measures real capability differences.

---

### Finding 4 — Logic Reasoning Improves Strongly With Scale

Logic accuracy improved substantially:

```text
Small: 0.200
Base:  0.333
Large: 0.667
```

Logic errors decreased:

```text
8 → 6 → 2
```

This suggests model scaling strongly improves basic symbolic reasoning on this benchmark.

---

### Finding 5 — Astronomy Knowledge Remains Weak

Astronomy accuracy stayed low:

```text
Small: 0.200
Base:  0.133
Large: 0.200
```

This suggests the FLAN-T5 family did not show strong astronomy-specific factual grounding on the current benchmark.

The benchmark therefore separates:

```text
general reasoning ability
```

from:

```text
domain-specific astronomy knowledge
```

---

### Finding 6 — Helpful Hints Do Not Always Help

FLAN-T5-Small performed worse with helpful hints:

```text
clean accuracy   = 0.300
helpful accuracy = 0.100
```

This suggests small instruction-tuned models may over-focus on or misinterpret supplemental context.

For Base and Large, helpful accuracy matched clean accuracy:

```text
helpful_hint_gain = 0.0
```

---

### Finding 7 — Misleading Hints Still Cause Failures

Even FLAN-T5-Large remained vulnerable to misleading hints.

Example:

```text
Question:
What force keeps planets in orbit around the Sun?

Clean prediction:
gravity

Misleading prediction:
magnetic force
```

This is a direct audit finding:

```text
The model knows the answer under clean conditions,
but can be pulled away by misleading information.
```

---

### Finding 8 — Larger Models Are Better But Not Fully Robust

FLAN-T5-Large was better overall and much better at logic, but it still had:

```text
7 misleading-hint errors
8 factual errors
```

Therefore:

```text
higher capability does not guarantee robustness or scientific grounding
```

---

## Important Qualitative Examples

### Example 1 — Misleading Hint Failure

Question:

```text
What force keeps planets in orbit around the Sun?
```

Ground truth:

```text
gravity
```

FLAN-T5-Large clean answer:

```text
gravity
```

FLAN-T5-Large misleading answer:

```text
magnetic force
```

Interpretation:

The model knows the correct answer, but misinformation in the prompt can override it.

---

### Example 2 — Logic Improvement With Scale

Question:

```text
If A is greater than B and B is greater than C, is A greater than C?
```

FLAN-T5-Large answer:

```text
A is greater than B. B is greater than C. The final answer: yes.
```

Interpretation:

The larger model showed better transitive reasoning than smaller FLAN-T5 models.

---

### Example 3 — Persistent Astronomy Weakness

Question:

```text
Which galaxy contains the Solar System?
```

Ground truth:

```text
Milky Way
```

Incorrect predictions included:

```text
Saturn
Neptune
Nebula
Andromeda Galaxy
```

Interpretation:

The FLAN-T5 family struggled with some astronomy-specific factual questions in the benchmark.

---

## Evaluation Limitations

### 1. Small Benchmark Size

The current benchmark has only:

```text
10 questions
30 prompt cases
```

This means one question can strongly affect metrics.

Future benchmark versions should expand to:

```text
50–100 questions
```

---

### 2. Limited Answer Matching

Current answer matching supports:

```text
normalized exact match
contains match
```

This is better than strict exact matching but still limited.

Future versions should support:

* answer aliases
* canonical answer maps
* semantic equivalence
* optional human review
* evaluator-model scoring

---

### 3. Synthetic Prompt Structure

The clean/helpful/misleading prompt format is simple.

Future work should test more prompt styles, such as:

```text
Question:
Hint:
Answer:
```

or structured JSON-like prompts.

---

### 4. Limited Model Coverage

Current real models are all from the FLAN-T5 family.

Future work should add:

```text
Qwen
Gemma
Phi
Llama-family models where feasible
```

This will make the audit more representative of modern LLM behavior.

---

### 5. No Faithfulness Analysis Yet

The current framework evaluates answer correctness and error type.

It does not yet evaluate whether the model's reasoning is faithful.

Future work should evaluate:

```text
whether the reasoning supports the answer
whether the model uses false hints in its rationale
whether the final answer matches the stated reasoning
```

---

## Current Project Status

The v1.0 framework now supports:

```text
benchmark loading
clean/helpful/misleading prompt generation
prompt-case expansion
synthetic model evaluation
real Hugging Face seq2seq model evaluation
answer matching
accuracy metrics
hint sensitivity metrics
error taxonomy
error distribution analysis
model comparison
visualization
experiment configs
test suite
```

This is a complete first version of a reasoning-audit framework.

---

## Suggested Version Meaning

```text
v1.0.0 = first complete reasoning-audit framework with real model validation
```

The v1.0 milestone is justified because the project has successfully evaluated:

```text
RuleBasedModel
SemiRobustModel
HintSensitiveModel
FLAN-T5-Small
FLAN-T5-Base
FLAN-T5-Large
```

and produced interpretable metrics, error distributions, and scaling behavior.

---

## Future Work

### v1.1 — Benchmark Expansion

Expand the benchmark from 10 questions to 50–100 questions.

Add categories:

```text
astronomy
physics
logic
arithmetic
scientific reasoning
observational reasoning
```

Goal:

```text
reduce noise and improve statistical reliability
```

---

### v1.2 — Modern Small LLMs

Evaluate modern lightweight models such as:

```text
Qwen
Gemma
Phi
```

Goal:

```text
compare older FLAN-T5 behavior with newer small language models
```

---

### v2.0 — Faithfulness Analysis

Move beyond correctness.

Add evaluation of:

```text
reasoning trace quality
answer-rationale consistency
hint usage
rationale faithfulness
```

Goal:

```text
determine whether the model's reasoning actually supports its answer
```

---

### v3.0 — Scientific Reasoning Audit

Develop a stronger science-focused benchmark with astronomy and physics reasoning.

Possible topics:

```text
redshift interpretation
spectral lines
light curves
planetary motion
observational uncertainty
evidence integration
scientific grounding
```

Goal:

```text
connect the audit framework more directly to scientific reasoning
```

---

## Final Interpretation

The main conclusion from v1.0 is:

```text
The reasoning-audit framework successfully distinguishes model capability,
robustness, and failure modes across synthetic and real models.
```

The strongest empirical finding is:

```text
FLAN-T5-Large substantially reduces logic errors compared to FLAN-T5-Small,
but astronomy-specific factual accuracy remains weak and misleading-hint failures persist.
```

This indicates that model scaling improves some reasoning capabilities, but does not automatically solve scientific grounding or robustness to misleading information.

The project is now ready to be committed and tagged as a complete v1.0 framework.
