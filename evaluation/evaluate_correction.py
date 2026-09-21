"""
evaluate_correction.py
----------------------
Measures CORRECTION performance: when the pipeline flags an answer, does the
self-correction loop actually produce a better answer?

Requires Ollama to be running (it generates real answers).

Reported metrics:
  * correction success rate  - share of questions whose FINAL answer is acceptable
  * repair rate              - share of initially-wrong answers that became acceptable
  * average correction iterations
  * average risk before and after correction

Run from the project root:
    python -m evaluation.evaluate_correction
"""

import argparse
import json
from pathlib import Path

from backend.ollama_client import OllamaClient, OllamaError
from backend.pipeline import run_pipeline
from backend.rag import knowledge_base

DEFAULT_FILE = Path(__file__).parent / "sample_data" / "correction_samples.json"

UNVERIFIABLE_MARKERS = ["could not be verified", "cannot be verified", "not verified",
                        "no information", "insufficient", "does not contain"]


def is_acceptable(answer: str, sample: dict) -> bool:
    lowered = answer.lower()
    if sample.get("expect_unverifiable"):
        return any(marker in lowered for marker in UNVERIFIABLE_MARKERS)
    if any(word.lower() in lowered for word in sample.get("forbid_keywords", [])):
        return False
    expected = sample.get("expect_keywords", [])
    return any(word.lower() in lowered for word in expected) if expected else True


def evaluate(path: Path) -> dict:
    samples = json.loads(path.read_text(encoding="utf-8"))["samples"]

    client = OllamaClient()
    status = client.health()
    if not status["running"] or not status["model_available"]:
        print("Cannot run the correction benchmark.")
        print(status["message"])
        return {}

    print("Building the knowledge base ...")
    knowledge_base.build()

    initial_ok = final_ok = repaired = 0
    iterations_total = 0
    risk_before_total = risk_after_total = 0

    for index, sample in enumerate(samples, start=1):
        print(f"\n[{index}/{len(samples)}] {sample['question']}")
        try:
            result = run_pipeline(sample["question"], knowledge_base, client)
        except OllamaError as exc:
            print(f"  skipped: {exc}")
            continue

        first_ok = is_acceptable(result["original_answer"], sample)
        last_ok = is_acceptable(result["final_answer"], sample)
        initial_ok += int(first_ok)
        final_ok += int(last_ok)
        repaired += int(last_ok and not first_ok)
        iterations_total += result["iterations"]
        risk_before_total += result["history"][0]["risk_score"]
        risk_after_total += result["risk_score"]

        print(f"  initial : {'OK ' if first_ok else 'BAD'} | risk {result['history'][0]['risk_score']}")
        print(f"  final   : {'OK ' if last_ok else 'BAD'} | risk {result['risk_score']} "
              f"after {result['iterations']} correction round(s)")
        print(f"  answer  : {result['final_answer'][:160]}")

    count = len(samples) or 1
    initially_wrong = count - initial_ok

    print("\n=== CORRECTION PERFORMANCE ===")
    print(f"questions                    : {count}")
    print(f"acceptable before correction : {initial_ok}/{count} ({initial_ok / count:.1%})")
    print(f"acceptable after correction  : {final_ok}/{count} ({final_ok / count:.1%})")
    if initially_wrong:
        print(f"repair rate                  : {repaired}/{initially_wrong} ({repaired / initially_wrong:.1%})")
    print(f"average correction rounds    : {iterations_total / count:.2f}")
    print(f"average risk before          : {risk_before_total / count:.1f}")
    print(f"average risk after           : {risk_after_total / count:.1f}")
    print("\nDetection performance is measured separately by evaluate_detection.py.")

    return {"initial_ok": initial_ok, "final_ok": final_ok, "repaired": repaired,
            "avg_iterations": iterations_total / count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognify-AI correction benchmark")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    args = parser.parse_args()
    evaluate(args.file)
