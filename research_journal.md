# Research Journal — Reasoning Audit

## 1. Project Purpose

The goal of this project was to build a reasoning-audit framework for studying how language models behave when the same question is presented under different prompt conditions:

```text
clean prompt
helpful-hint prompt
misleading-hint prompt
```

The project was not only about measuring accuracy. It was about separating several different behaviors:

```text
task correctness
usefulness of helpful hints
vulnerability to misleading hints
domain-specific weaknesses
reliability of the evaluator itself
```

Over the course of the project, the central focus shifted from simply evaluating models to building a more trustworthy evaluation method.

---

## 2. Initial Free-Form Benchmark

The first version used free-form answers.

Each benchmark row contained:

```text
question
answer
helpful_hint
misleading_hint
```

Each question was expanded into three prompt cases:

```text
clean
helpful
misleading
```

This gave the first complete pipeline:

```text
CSV benchmark
→ prompt generation
→ model response
→ answer matching
→ metrics
→ error analysis
```

The early version was useful because it tested the full system end to end. It also exposed the most important weakness in the project: free-form answer matching was unreliable.

---

## 3. Synthetic Model Validation

Before evaluating real models, synthetic models were added.

These included:

```text
MockModel
RuleBasedModel
HintSensitiveModel
SemiRobustModel
```

This was an important step because it allowed the evaluation pipeline to be tested under known behaviors.

The synthetic phase confirmed that the framework could distinguish:

```text
a model that knows the answer
a model that follows misleading hints
a model that is partially robust
```

The main lesson was:

```text
Correctness and robustness are not the same thing.
```

A model can answer clean prompts correctly and still fail when misleading context is introduced.

---

## 4. First Real Model Experiments

The first real experiments used FLAN-T5 models.

These tests showed that the pipeline could run real Hugging Face models and produce useful prompt-type metrics.

However, manual inspection revealed that many apparent model errors were actually evaluator errors.

For example, a model could give a semantically correct answer but still be marked wrong because the wording differed from the expected answer.

This made it clear that the project needed stronger evaluation design, not just more models.

---

## 5. Free-Form Evaluation Failure

The largest issue was answer matching.

Examples of false negatives:

```text
0.25 vs 1/4
constant velocity vs velocity becomes constant
changing Sun-Earth-Moon geometry vs relative positions of the Sun, Earth, and Moon
```

Examples of possible false positives:

```text
ground truth: 0.5
prediction: -0.5
```

A simple substring matcher can incorrectly count the second case as correct.

I tried improving the matcher with aliases, normalization, and numeric matching. This helped, but it did not solve the deeper problem.

The conclusion was:

```text
Free-form evaluation was too ambiguous for reliable scoring.
```

This became the turning point of the project.

---

## 6. Redesign to Multiple Choice

The benchmark was redesigned into multiple-choice format.

Instead of requiring the evaluator to judge semantic equivalence, the model now had to answer with:

```text
A
B
C
D
```

This made evaluation objective:

```text
prediction == answer_key
```

The MCQ redesign was the most important methodological improvement in the project.

It changed the benchmark from a fragile free-form scoring problem into a reproducible option-letter evaluation task.

The MCQ format also made manual inspection easier. When a model failed, the failure was clear.

---

## 7. MCQ Pilot and Full Benchmark

A small MCQ pilot was created first to test:

```text
MCQ prompt formatting
option-letter extraction
metrics
error taxonomy
model behavior under hints
```

After the pilot worked, the full MCQ benchmark was created.

The final benchmark contains:

```text
100 questions
25 astronomy
25 logic
25 physics
25 arithmetic
```

The answer keys were balanced:

```text
A: 25
B: 25
C: 25
D: 25
```

A held-out split was then generated:

```text
80 training questions
20 test questions
```

The test set was balanced across both domains and answer letters:

```text
5 questions per domain
5 answers per option letter
```

This made the final benchmark significantly more reliable than the earlier free-form versions.

---

## 8. LoRA Fine-Tuning

LoRA fine-tuning was added for FLAN-T5-Base.

The purpose was to test whether small-scale adaptation improves held-out reasoning performance.

The final MCQ LoRA setup trained on:

```text
80 MCQ training questions
```

and evaluated on:

```text
20 held-out MCQ test questions
```

The training data was updated so that MCQ rows became instruction-style examples:

```text
input: question + answer options
target: correct option letter
```

This made the fine-tuning setup consistent with the final evaluation format.

---

## 9. Final Model Comparison

The final comparison used:

```text
FLAN-T5-Base
FLAN-T5-Base + MCQ LoRA
Qwen2.5-0.5B-Instruct
```

Final MCQ test results:

| Model                   | Overall | Clean | Helpful | Misleading |
| ----------------------- | ------: | ----: | ------: | ---------: |
| FLAN-T5-Base            |   0.300 | 0.300 |   0.400 |      0.200 |
| FLAN-T5-Base + MCQ LoRA |   0.317 | 0.350 |   0.400 |      0.200 |
| Qwen2.5-0.5B-Instruct   |   0.350 | 0.450 |   0.350 |      0.250 |

Final ranking:

```text
1. Qwen2.5-0.5B-Instruct
2. FLAN-T5-Base + MCQ LoRA
3. FLAN-T5-Base
```

The LoRA adapter slightly improved FLAN-T5-Base overall accuracy:

```text
0.300 → 0.317
```

and clean accuracy:

```text
0.300 → 0.350
```

However, misleading accuracy did not improve:

```text
0.200 → 0.200
```

This was one of the most important findings.

---

## 10. Interpretation

The final results suggest that task adaptation and robustness are different.

LoRA helped FLAN-T5-Base slightly on clean performance, but it did not make the model more resistant to misleading hints.

Qwen2.5-0.5B performed best overall, but it also showed sensitivity to prompt context. Its helpful accuracy was lower than its clean accuracy:

```text
clean:   0.450
helpful: 0.350
```

This suggests that extra context can distract small instruction models, even when that context is intended to help.

Across the final models, misleading prompts generally reduced performance. This confirms that the benchmark is measuring a real robustness issue.

---

## 11. Main Lessons

### Evaluation design matters

The biggest lesson was that model scores are only meaningful if the evaluator is reliable.

The free-form benchmark produced ambiguous results because scoring depended too much on wording.

The MCQ redesign made the evaluation cleaner and more auditable.

### Robustness is not the same as accuracy

A model can improve on clean prompts without improving under misleading prompts.

This happened with MCQ LoRA.

### Helpful hints are not guaranteed to help

Helpful hints improved FLAN-T5-Base but reduced Qwen performance.

This suggests that prompt context can interact with model behavior in non-obvious ways.

### Manual inspection is necessary

Manual review of model outputs revealed evaluator problems that aggregate metrics alone would have hidden.

This was crucial for improving the benchmark.

---

## 12. Technical Notes

Several practical issues were encountered.

A newer PEFT version caused compatibility problems, so the stable setup used:

```text
transformers==4.44.2
peft==0.12.0
```

Qwen2.5-1.5B was attempted but was too memory-heavy for comfortable local evaluation, so the final Qwen experiments used:

```text
Qwen2.5-0.5B-Instruct
```

The `.gitignore` initially ignored all CSV files, which accidentally excluded benchmark data. This was corrected so benchmark source files are tracked, while generated train/test splits remain ignored.

---

## 13. What I Would Do Differently

If starting again, I would use MCQ evaluation from the beginning.

The free-form stage was still useful because it revealed the evaluation problem, but it created avoidable complexity.

A better workflow would be:

```text
1. Build MCQ benchmark first
2. Balance answer keys early
3. Create train/test split early
4. Validate with synthetic models
5. Evaluate real models
6. Fine-tune only after the benchmark is stable
```

I would also keep faithfulness evaluation separate from answer correctness from the start.

---

## 14. Future Direction

The current project evaluates final-answer correctness under prompt perturbations.

A future extension could evaluate reasoning faithfulness by asking models to output structured answers such as:

```json
{
  "answer": "B",
  "reason": "The visible phase depends on the relative positions of the Sun, Earth, and Moon."
}
```

Then the evaluator could score:

```text
answer correctness
reasoning correctness
answer-reason consistency
use of misleading hints
```

This would be a natural next project, but it should remain separate from the current benchmark to avoid mixing answer accuracy with rationale evaluation.

---

## 15. Final Conclusion

The project developed from a simple free-form reasoning benchmark into a more reliable MCQ-based reasoning-audit framework.

The central methodological conclusion is:

```text
Free-form answer matching was too ambiguous for reliable scoring.
MCQ option-letter evaluation produced a more trustworthy benchmark.
```

The central empirical conclusion is:

```text
Qwen2.5-0.5B-Instruct achieved the best final MCQ performance.
```

The central robustness conclusion is:

```text
MCQ LoRA slightly improved FLAN-T5-Base accuracy, but did not improve robustness to misleading hints.
```

Overall, the project shows that reasoning evaluation requires more than accuracy. A good audit must also test context sensitivity, misleading-hint vulnerability, and the reliability of the scoring method itself.
