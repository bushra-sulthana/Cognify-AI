# Evaluation

Two separate things are measured, and they must not be mixed up:

| Script | Question it answers | Metrics |
|---|---|---|
| `evaluate_detection.py` | Does the verifier correctly flag unsupported claims? | accuracy, precision, recall, F1 |
| `evaluate_correction.py` | Does the self-correction loop actually fix flagged answers? | correction success rate, repair rate, average iterations, risk before/after |

## Run (from the project root, with the virtual environment active)

```bat
python -m evaluation.evaluate_detection
python -m evaluation.evaluate_correction
```

`evaluate_detection.py` needs no LLM. `evaluate_correction.py` needs Ollama running.

## Using TruthfulQA, FEVER or HaluEval

No dataset is downloaded automatically, because these datasets are large and their
licences differ. Download a slice yourself and convert it into the format used in
`sample_data/detection_samples.json`:

```json
{"samples": [{"claim": "<one sentence>", "label": "SUPPORTED | NOT_SUPPORTED"}]}
```

* **FEVER** – `claim` maps directly; `SUPPORTS` → `SUPPORTED`, `REFUTES` and `NOT ENOUGH INFO` → `NOT_SUPPORTED`.
* **TruthfulQA** – use `best_answer` as a SUPPORTED claim and one entry from `incorrect_answers` as NOT_SUPPORTED.
* **HaluEval** – `right_answer` → SUPPORTED, `hallucinated_answer` → NOT_SUPPORTED.

Then run:

```bat
python -m evaluation.evaluate_detection --file evaluation\sample_data\my_fever_slice.json
```

**Important:** the evidence the verifier can see is limited to `knowledge_base/`.
If you evaluate FEVER claims about topics your knowledge base does not cover,
everything will be reported as `INSUFFICIENT` and recall will look artificially
perfect while precision collapses. Always add the matching source documents to
`knowledge_base/` before benchmarking, and report which knowledge base was used.
