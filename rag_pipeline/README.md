# RAG Pipeline over Annuity Product Documents

This is Part B of the `llm_evaluation` project. See the top-level
`README.md` for how this relates to the Part A LLM evaluation work.

## What this is

A small, complete retrieval-augmented generation (RAG) pipeline built
over a set of illustrative variable annuity reference documents (product
overview, surrender charge schedule, and guaranteed living benefit
riders). It demonstrates the core RAG mechanics end to end: document
chunking, embedding, vector storage, semantic retrieval, and grounded
LLM generation.

This project was built to gain direct, hands-on experience with the RAG
components referenced in the Part A evaluation work, after completing a
LangChain course and wanting an artifact built and understood from
scratch rather than a tutorial walkthrough.

## A known, intentional limitation

The embedding step uses a TF-IDF vectorizer rather than a neural
embedding model, because the build environment used to develop this
project could not reach external model hosts to download one. This is
documented in detail in `src/build_index.py`, including a worked example
showing where TF-IDF retrieval succeeds (queries that share vocabulary
with the source text) and where it fails (paraphrased queries with
different wording but the same meaning) -- the exact failure mode that
neural embeddings are designed to solve. Swapping in a real embedding
model (e.g. `sentence-transformers/all-MiniLM-L6-v2` or
`text-embedding-3-small`) is a one-line change once a model download or
API key is available; the rest of the pipeline is unaffected.

## Repository structure

```text
rag_pipeline/
|
+-- data/
|   +-- product_overview.txt
|   +-- surrender_charges.txt
|   +-- living_benefit_riders.txt
|
+-- src/
|   +-- ingest.py        # load + chunk documents
|   +-- build_index.py   # embed chunks, build FAISS vector store
|   +-- query.py         # retrieve + generate grounded answers
|
+-- requirements.txt
+-- README.md
```

## How to run

```bash
pip install -r requirements.txt
python src/ingest.py        # inspect chunking
python src/build_index.py   # build the vector index, run a retrieval sanity check
python src/query.py         # run example questions end to end
```

`query.py` expects an OpenAI-compatible LLM endpoint (configured for a
local server via LM Studio by default, matching the Part A evaluation
setup) for the generation step. The retrieval steps run independently of
this and will print results even without a running model server.

## Example query flow

Question: *"What percentage surrender charge applies in year 3 of the
contract?"*

1. The question is embedded and compared against all stored chunk
   vectors.
2. The top-k most similar chunks are retrieved -- in this case, the
   surrender charge percentage table itself.
3. Retrieved chunks are inserted into a prompt that instructs the model
   to answer only from the provided context.
4. The LLM returns an answer grounded in that retrieved text, rather
   than from general training knowledge.

## Future work

- Swap TF-IDF for a real neural embedding model.
- Add a retrieval evaluation step (e.g. does the correct source chunk
  appear in the top-k results) rather than only spot-checking outputs
  manually.
- Compare chunking strategies (fixed-size vs. semantic/section-based
  splitting) and measure the effect on retrieval quality.
- Extend to a larger, more realistic document set.
