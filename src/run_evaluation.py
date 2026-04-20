import json
from pathlib import Path

from openai import OpenAI
from utils import load_yaml, ensure_dir
from metrics import compute_accuracy, summarize_uncertainty_language


def llm_call(prompt: str, config: dict) -> dict:
    client = OpenAI(
        base_url=config["model"]["base_url"],
        api_key="lm-studio"
    )

    response = client.chat.completions.create(
        model=config["model"]["model_name"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config["model"]["max_tokens"],
        temperature=config["model"]["temperature"],
    )

    raw = response.choices[0].message.content
    print("\nRAW MODEL OUTPUT:\n", raw, "\n")

    cleaned = raw.strip()

    # Remove markdown fences
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1].strip()

    # Remove leading "json"
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    # Try parsing JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "chosen_value": None,
            "rationale": "Model returned non-JSON output.",
            "uncertainty_description": raw or ""
        }


def build_prompt(sample: dict, base_prompts: dict) -> str:
    task_type = sample["type"]

    # --- Conflicting Sources Task ---
    if task_type == "conflicting_sources":
        prompt = base_prompts["conflicting_sources"]
        return prompt.format(
            value_a=sample["sources"]["A"],
            value_b=sample["sources"]["B"],
            value_c=sample["sources"]["C"]
        )

    # --- Missing Data Task ---
    elif task_type == "missing_data":
        prompt = base_prompts["missing_data"]
        known_fields_str = json.dumps(sample["observed_record"], indent=2)
        return prompt.format(known_fields=known_fields_str)

    # --- Unknown Task Type ---
    raise ValueError(f"Unknown sample type: {task_type}")



def load_prompts(path: str) -> dict:
    prompts = {}
    current_key = None
    current_lines = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")

            # Detect section header
            if stripped.startswith("===") and stripped.endswith("==="):
                # Save previous section
                if current_key is not None:
                    prompts[current_key] = "\n".join(current_lines).strip()

                # Start new section
                current_key = stripped.strip("=").strip()
                current_lines = []
            else:
                current_lines.append(stripped)

    # Save last section
    if current_key is not None:
        prompts[current_key] = "\n".join(current_lines).strip()

    return prompts


def main():
    config = load_yaml("config/config.yaml")
    processed_path = Path(config["data"]["processed_dir"]) / "evaluation_samples.jsonl"
    outputs_path = Path(config["evaluation"]["output_path"])
    summary_path = Path(config["evaluation"]["summary_path"])

    ensure_dir(outputs_path.parent)
    ensure_dir(summary_path.parent)

    base_prompts = load_prompts("prompts/base_prompts.txt")
    print(base_prompts.keys())

    records = []
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            print("Sample:", sample)
            prompt = build_prompt(sample, base_prompts)
            model_output = llm_call(prompt, config)
            records.append(
                {
                    "sample_id": sample["id"],
                    "sample": sample,
                    "prompt": prompt,
                    "model_output": model_output,
                }
            )

    outputs_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    acc = compute_accuracy(records)
    unc = summarize_uncertainty_language(records)

    summary = f"""
# LLM Reliability Evaluation — Summary

- Samples: {len(records)}
- Accuracy (conflicting sources): {acc:.3f}
- Fraction with uncertainty: {unc['fraction_with_uncertainty']:.3f}
- Avg uncertainty word count: {unc['avg_uncertainty_word_count']:.2f}

Note: Replace with real model comparisons as needed.
"""

    summary_path.write_text(summary.strip(), encoding="utf-8")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
