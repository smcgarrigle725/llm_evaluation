"""
Step 3: Retrieval-Augmented Generation -- the full pipeline.

WHAT "AUGMENTED GENERATION" ACTUALLY MEANS
A plain LLM answers purely from what it learned during training. That
creates two problems for a use case like this: the model has no idea what
THIS specific insurance product's terms are (they were never in its
training data), and even for general insurance knowledge, the model could
confidently state something subtly wrong (hallucinate) with no way for us
to check it.

RAG fixes this by changing what we ask the model to do. Instead of "answer
this question from what you know," the prompt becomes "here are some
retrieved passages from real source documents -- answer the question using
ONLY this information." This grounds the model's answer in retrievable,
checkable text, and lets us cite exactly which document the answer came
from.

THE FOUR STEPS THAT HAPPEN ON EVERY QUERY
1. Embed the user's question using the SAME embedding method used to
   embed the documents (this matters -- query and documents must live in
   the same vector space to be comparable).
2. Retrieve the k most similar chunks from the vector store.
3. Insert those chunks into a prompt template alongside the question.
4. Send that combined prompt to an LLM and return its answer.

WHY THIS LOCAL LLM SETUP?
This script calls a local model via LM Studio (an OpenAI-compatible local
server), the same pattern used in the LLM evaluation project. This avoids
needing an API key for this demo. The exact same code works against
OpenAI, Anthropic, or any other provider by changing the base_url and
model name -- the RAG logic itself (steps 1-4 above) is identical
regardless of which model answers the final question.
"""

import sys
from pathlib import Path

from langchain_community.vectorstores import FAISS
from openai import OpenAI

from build_index import TfidfEmbeddings, build_vector_store, INDEX_DIR

PROMPT_TEMPLATE = """Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say so
explicitly rather than guessing.

Context:
{context}

Question: {question}

Answer:"""


def load_or_build_index():
    """FAISS indexes saved with a custom Embeddings subclass need that
    exact class available to deserialize, so we rebuild fresh here rather
    than loading from disk -- simpler and fast enough at this small scale.
    A production system would persist a fitted, reusable embedding model
    instead of refitting on every run."""
    return build_vector_store()


def retrieve(vector_store, question: str, k: int = 3):
    return vector_store.similarity_search(question, k=k)


def format_context(chunks) -> str:
    """Join retrieved chunks into a single context block, labeling each
    with its source document. Labeling matters: it's what lets the final
    answer (or a human reviewer) trace a claim back to a specific source,
    which is the whole point of grounding generation in retrieval."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk.metadata['source']}]\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate_answer(question: str, context: str, base_url: str = "http://localhost:1234/v1") -> str:
    client = OpenAI(base_url=base_url, api_key="lm-studio")
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic answers for a Q&A task -- we
                          # want consistent factual retrieval, not creative
                          # variation, so temperature is set to 0
    )
    return response.choices[0].message.content


def answer_question(vector_store, question: str, k: int = 3, call_llm: bool = True):
    chunks = retrieve(vector_store, question, k=k)
    context = format_context(chunks)

    print(f"\nQuestion: {question}")
    print(f"\nRetrieved {len(chunks)} chunks:")
    for c in chunks:
        print(f"  - [{c.metadata['source']}] {c.page_content[:80]}...")

    if call_llm:
        try:
            answer = generate_answer(question, context)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"\n(LLM call failed -- no local model server running: {e})")
            print("Retrieval succeeded; generation step requires a running LLM endpoint.")
    return chunks


if __name__ == "__main__":
    vector_store = load_or_build_index()

    test_questions = [
        "What percentage surrender charge applies in year 3 of the contract?",
        "How does the GMWB benefit base grow if no withdrawals are taken?",
    ]

    for q in test_questions:
        answer_question(vector_store, q, k=2, call_llm=False)
        print("\n" + "=" * 70)
