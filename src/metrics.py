from typing import List, Dict, Any
import statistics


def compute_accuracy(records: List[Dict[str, Any]]) -> float:
    correct = 0
    total = 0
    for r in records:
        if r["sample"]["type"] != "conflicting_sources":
            continue
        total += 1
        true_val = r["sample"]["true_value"]
        pred = r["model_output"].get("chosen_value")
        if pred == true_val:
            correct += 1
    return correct / total if total else 0.0


def summarize_uncertainty_language(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = []
    for r in records:
        desc = r["model_output"].get("uncertainty_description", "")
        if desc.strip():
            lengths.append(len(desc.split()))
    return {
        "fraction_with_uncertainty": len(lengths) / len(records),
        "avg_uncertainty_word_count": statistics.mean(lengths) if lengths else 0.0,
    }
