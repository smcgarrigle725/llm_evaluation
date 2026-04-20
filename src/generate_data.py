import json
import random
from pathlib import Path

from utils import load_yaml, ensure_dir


def generate_conflicting_sources(n: int, noise_level: str):
    samples = []
    for i in range(n):
        true_value = random.randint(10, 100)
        a = true_value
        b = true_value + random.choice([-5, 0, 5])
        c = true_value + random.choice([-10, 0, 10])

        samples.append(
            {
                "id": f"conflict_{noise_level}_{i}",
                "type": "conflicting_sources",
                "true_value": true_value,
                "sources": {"A": a, "B": b, "C": c},
                "noise_level": noise_level,
            }
        )
    return samples


def generate_missing_data(n: int, noise_level: str):
    samples = []
    for i in range(n):
        base = {
            "age": random.randint(18, 80),
            "income": random.randint(30_000, 150_000),
            "city": random.choice(["Boston", "Chicago", "Seattle"]),
        }
        field_to_drop = random.choice(list(base.keys()))
        observed = {k: v for k, v in base.items() if k != field_to_drop}

        samples.append(
            {
                "id": f"missing_{noise_level}_{i}",
                "type": "missing_data",
                "full_record": base,
                "observed_record": observed,
                "missing_field": field_to_drop,
                "noise_level": noise_level,
            }
        )
    return samples


def main():
    config = load_yaml("config/config.yaml")
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)

    n = config["data"]["n_samples"]
    noise_levels = config["data"]["noise_levels"]

    all_samples = []
    for level in noise_levels:
        all_samples.extend(generate_conflicting_sources(n // 2, level))
        all_samples.extend(generate_missing_data(n // 2, level))

    raw_path = raw_dir / "synthetic_samples.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    processed_path = processed_dir / "evaluation_samples.jsonl"
    processed_path.write_text(raw_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Wrote {len(all_samples)} samples.")


if __name__ == "__main__":
    main()
