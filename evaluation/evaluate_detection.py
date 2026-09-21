"""
evaluate_detection.py
---------------------
Measures DETECTION performance only: given a claim, does Cognify-AI correctly
decide whether the local knowledge base supports it?

Positive class = "flagged as a possible hallucination"
                 (verification status is CONTRADICTED or INSUFFICIENT).

Metrics: accuracy, precision, recall, F1, plus the confusion matrix.

Run from the project root:
    python -m evaluation.evaluate_detection
    python -m evaluation.evaluate_detection --file evaluation/sample_data/detection_samples.json

This script does NOT download any dataset. To evaluate on TruthfulQA, FEVER or
HaluEval, download them yourself and convert a small slice into the same JSON
format (see evaluation/README.md).
"""

import argparse
import json
from pathlib import Path

from backend.rag import knowledge_base
from backend.verifier import SUPPORTED, verify_claim

DEFAULT_FILE = Path(__file__).parent / "sample_data" / "detection_samples.json"


def evaluate(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload["samples"] if isinstance(payload, dict) else payload

    print(f"Building the knowledge base from {knowledge_base.folder} ...")
    knowledge_base.build()
    print(f"Retrieval backend: {knowledge_base.backend}\n")

    tp = fp = tn = fn = 0
    rows = []

    for sample in samples:
        result = verify_claim(sample["claim"], knowledge_base)
        flagged = result["status"] != SUPPORTED              # predicted hallucination
        should_flag = sample["label"] != "SUPPORTED"          # ground truth

        if flagged and should_flag:
            tp += 1; outcome = "TP"
        elif flagged and not should_flag:
            fp += 1; outcome = "FP"
        elif not flagged and not should_flag:
            tn += 1; outcome = "TN"
        else:
            fn += 1; outcome = "FN"

        rows.append((outcome, result["status"], sample["label"], sample["claim"]))

    total = tp + fp + tn + fn or 1
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"{'OUTCOME':<8}{'PREDICTED':<15}{'TRUTH':<16}CLAIM")
    print("-" * 100)
    for outcome, predicted, truth, claim in rows:
        print(f"{outcome:<8}{predicted:<15}{truth:<16}{claim[:60]}")

    print("\n=== DETECTION PERFORMANCE ===")
    print(f"samples            : {total}")
    print(f"true positives     : {tp}   (hallucination correctly flagged)")
    print(f"false positives    : {fp}   (correct claim wrongly flagged)")
    print(f"true negatives     : {tn}")
    print(f"false negatives    : {fn}   (hallucination missed)")
    print(f"accuracy           : {accuracy:.3f}")
    print(f"precision          : {precision:.3f}")
    print(f"recall             : {recall:.3f}")
    print(f"F1 score           : {f1:.3f}")
    print("\nNote: these numbers describe detection on THIS small sample against THIS")
    print("knowledge base. They are not a general benchmark result.")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognify-AI detection benchmark")
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="JSON file with claims")
    args = parser.parse_args()
    evaluate(args.file)
