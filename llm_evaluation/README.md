# LLM Reliability Under Noisy, Multi-Source Data

## Overview

Real-world data is rarely clean or consistent. In many applied settings, information is incomplete, conflicting, or contains irrelevant signals. While large language models (LLMs) perform well on structured benchmarks, their behavior under these more realistic conditions is less well understood.

This project evaluates how LLMs respond to noisy, multi-source inputs, with a focus on identifying failure modes related to hallucination, overconfidence, and sensitivity to irrelevant or conflicting information.

The goal is to better understand how LLMs behave in conditions that more closely resemble real-world data environments.

---

## Research Questions

This project explores the following questions:

- How do LLMs handle conflicting information across multiple sources?
- What happens when inputs contain missing or incomplete data?
- Are LLMs sensitive to irrelevant or misleading features?
- Do models exhibit overconfidence under uncertainty?
- When do models hallucinate or fabricate information to fill gaps?

---

## Approach

To study these behaviors, I construct synthetic but realistic datasets that simulate common issues in real-world data pipelines:

- Multiple sources with conflicting values  
- Missing or null fields  
- Irrelevant or noisy features  
- Slightly misleading context or framing  

These datasets are paired with structured prompts and evaluated using a combination of qualitative analysis and simple quantitative metrics.

The evaluation pipeline includes:

1. Generating noisy, multi-source inputs  
2. Prompting an LLM to perform reasoning or estimation tasks  
3. Capturing model outputs and reasoning patterns  
4. Analyzing correctness, consistency, and confidence  

---

## Repository Structure

```text
llm_evaluation/
|
+-- config/
|   +-- config.yaml
|
+-- data/
|   +-- raw/
|   +-- processed/
|
+-- notebooks/
|   +-- exploratory_analysis.ipynb
|
+-- prompts/
|   +-- base_prompts.txt
|   +-- adversarial_prompts.txt
|
+-- results/
|   +-- outputs.json
|   +-- summary.md
|
+-- src/
|   +-- generate_data.py
|   +-- run_evaluation.py
|   +-- metrics.py
|   +-- utils.py
|
+-- README.md
+-- requirements.txt
```

---

## Key Experiments

### 1. Conflicting Sources
Evaluate how models reconcile multiple sources that provide different answers.

### 2. Missing Data
Test whether models acknowledge uncertainty or fabricate missing information.

### 3. Irrelevant Features
Introduce distracting or unrelated variables to test robustness.

---

## Preliminary Observations

(*To be updated as experiments are completed*)

Early results suggest that:

- Models may overweight a single source, even when multiple are provided  
- Models often fail to explicitly represent uncertainty when data is incomplete  
- Irrelevant features can sometimes influence outputs in non-obvious ways  
- Outputs may remain confident even when underlying data is ambiguous or conflicting  

---

## Why This Matters

Many real-world AI applications involve:

- Integrating multiple data sources  
- Reasoning under uncertainty  
- Making decisions with incomplete information  

Understanding how LLMs behave in these conditions is important for:

- Improving model reliability and robustness  
- Designing better evaluation frameworks  
- Informing safer deployment of AI systems  

---

## How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Generate synthetic data:

   ```bash
   python src/generate_data.py
   ```

3. Run evaluation:

   ```bash
   python src/run_evaluation.py
   ```

4. Analyze results:

   - Open `notebooks/exploratory_analysis.ipynb`
   - Review `results/summary.md`

---

## Future Work

- Expand evaluation to additional model families  
- Develop more formal metrics for uncertainty and confidence  
- Explore mitigation strategies for identified failure modes  
- Extend to real-world datasets with known inconsistencies  

---

*llm_evaluation — Samantha McGarrigle*