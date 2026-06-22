"""
Step 2: Embed chunks and store them in a vector database.

WHAT IS AN EMBEDDING, CONCRETELY?
An embedding model converts a piece of text into a list of numbers (a
vector) -- in this case, 384 numbers per chunk. The model is trained so
that chunks of text with similar MEANING end up as vectors that are close
together in that 384-dimensional space, even if they don't share any of
the same words. This is what lets retrieval work by semantic similarity
("withdraw money early" can match a chunk about "surrender charges" even
though the words don't overlap) rather than by exact keyword matching.

WHICH EMBEDDING MODEL, AND WHY?
This build environment cannot reach huggingface.co to download a neural
embedding model (sandboxed network), so this script uses a TF-IDF based
embedding instead -- implemented below as a small custom class that
satisfies LangChain's Embeddings interface. TF-IDF turns each chunk into a
vector based on word frequency, weighted so that words common across all
documents count for less than words that are distinctive to a particular
chunk.

This is a meaningfully weaker substitute for a real semantic embedding
model: TF-IDF only matches on shared vocabulary, not on meaning, so a
query like "take money out early" will NOT reliably match a chunk about
"surrender charges" the way a true neural embedding model would, since
those phrases share almost no words. A production RAG system (and the
honest answer in an interview) should use a real neural embedding model
such as sentence-transformers/all-MiniLM-L6-v2, OpenAI's
text-embedding-3-small, or a similar model -- the LangChain interface is
identical, so swapping this class for HuggingFaceEmbeddings or
OpenAIEmbeddings is a one-line change once a model download or API key is
available.

WHAT IS A VECTOR DATABASE, AND WHY FAISS?
A vector database stores these embeddings and is built to answer one
specific kind of query fast: "given this query vector, which stored
vectors are closest to it?" That's a different problem from a normal SQL
database, which isn't built to efficiently search by numerical closeness
across hundreds of dimensions.

FAISS (Facebook AI Similarity Search) is a lightweight, local vector
index -- it runs in-process, with no separate server to stand up, which
makes it a reasonable choice for a small demo project. Production systems
often use a managed vector database (Pinecone, Weaviate, pgvector, etc.)
instead, mainly for persistence, scaling past memory limits, and
multi-user concurrent access -- the underlying nearest-neighbor search
concept is the same.
"""

from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer

from ingest import load_documents, split_documents

INDEX_DIR = Path(__file__).parent.parent / "faiss_index"


class TfidfEmbeddings(Embeddings):
    """A minimal embedding class that fits LangChain's Embeddings interface
    (embed_documents / embed_query) but is backed by TF-IDF instead of a
    downloaded neural network. This exists ONLY because this sandbox can't
    reach huggingface.co -- it is explicitly a fallback, not a recommended
    production approach. See the module docstring above for why."""

    def __init__(self, max_features: int = 384):
        # max_features=384 chosen to roughly match the dimensionality of
        # a small neural embedding model, purely so the rest of the
        # pipeline (FAISS index, similarity search) behaves identically
        # regardless of which embedding backend is plugged in.
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    def fit(self, texts: List[str]):
        self.vectorizer.fit(texts)
        self._fitted = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.vectorizer.transform([text]).toarray()[0].tolist()


def build_vector_store():
    docs = load_documents()
    chunks = split_documents(docs)

    print(f"Embedding {len(chunks)} chunks with TF-IDF "
          f"(fallback embedding -- see module docstring)...")

    embeddings = TfidfEmbeddings(max_features=384)
    # Fit the vectorizer's vocabulary on our actual chunk text before
    # embedding, since TF-IDF (unlike a pretrained neural model) has no
    # vocabulary until it has seen the corpus it will be used on.
    embeddings.fit([c.page_content for c in chunks])

    # This is the actual embedding + indexing step: every chunk gets
    # converted to a vector and added to the FAISS index.
    vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(str(INDEX_DIR))
    print(f"Saved FAISS index to {INDEX_DIR}")
    return vector_store


if __name__ == "__main__":
    vector_store = build_vector_store()

    # Quick sanity check that retrieval works at all. Note: with TF-IDF,
    # the query needs to share actual words with the source text to match
    # well -- this is the limitation called out in the module docstring.
    test_query = "surrender charge withdrawal early years"
    results = vector_store.similarity_search(test_query, k=2)

    print(f"\nTest query: '{test_query}'")
    print("Top matches:")
    for r in results:
        print(f"\n  [{r.metadata['source']}]")
        print(f"  {r.page_content[:200]}...")
