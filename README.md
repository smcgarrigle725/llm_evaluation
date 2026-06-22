# LLM Reliability and Retrieval-Augmented Generation

This project investigates two related but distinct aspects of working
with large language models in applied settings: how reliably they
reason over noisy or incomplete information (**Part A**), and how
retrieval grounding can be used to reduce hallucination and anchor
answers in real source documents (**Part B**).

---

## Part A: LLM Reliability Under Noisy, Multi-Source Data

**Location:** `llm_evaluation/`

This is an evaluation harness that tests how an LLM behaves when given
conflicting information from multiple sources, or records with missing
fields. It measures whether the model resolves conflicts correctly,
whether it appropriately expresses uncertainty rather than guessing, and
analyzes qualitative patterns in hallucination and overconfidence.

Note: this is direct LLM prompting and evaluation -- it does not use
retrieval or a vector database. See Part B below for the retrieval
component.

See `llm_evaluation/README.md` for full details, research questions, and
how to run it.

---

## Part B: RAG Pipeline over Annuity Product Documents

**Location:** `rag_pipeline/`

A complete retrieval-augmented generation pipeline built over a set of
illustrative insurance/annuity reference documents. This project was
built to gain direct, hands-on experience with retrieval-augmented
generation, vector embeddings, and LangChain after completing coursework
in the area -- the goal was to build and understand a working pipeline
end to end, rather than only completing tutorial exercises.

It covers the full RAG flow: document chunking, embedding, vector
storage (FAISS), semantic retrieval, and grounded LLM generation, with
the reasoning behind each design decision documented directly in the
code. It also includes a worked demonstration of a real embedding
limitation (TF-IDF vs. neural embeddings) encountered while building it,
documented in `rag_pipeline/src/build_index.py`.

See `rag_pipeline/README.md` for full details, structure, and how to
run it.

---

## How Part A and Part B relate

Both parts are ultimately about the same underlying question: how do we
get trustworthy, well-grounded answers out of an LLM rather than
plausible-sounding but unverified ones? Part A studies failure modes in
unconstrained reasoning over messy data. Part B is one concrete mitigation
strategy -- grounding generation in retrieved, citable source text -- and
demonstrates that mitigation hands-on.

## Future Work

- Apply the Part A evaluation methodology (conflict resolution,
  uncertainty expression, hallucination detection) directly to the Part B
  RAG pipeline's outputs, to measure whether grounding actually reduces
  the failure rates observed in Part A.
- Swap the Part B fallback TF-IDF embeddings for a neural embedding model
  and compare retrieval quality.
- Expand both components to additional model families and larger,
  more realistic datasets.

---

*llm_evaluation — Samantha McGarrigle*
