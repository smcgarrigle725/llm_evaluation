"""
Step 1: Load documents and split them into chunks.

WHY CHUNK AT ALL?
Embedding models and LLMs both have limits on how much text they can
usefully process at once. More importantly, if we embedded each entire
document as a single vector, a question about one specific detail (like
the free withdrawal percentage) would get diluted by everything else in
that document. Smaller, focused chunks mean each embedded vector
represents a narrower, more specific piece of meaning, which improves
retrieval precision.

WHY THESE SPECIFIC PARAMETERS?
- chunk_size=500: roughly 100-125 words per chunk. Small enough to stay
  topically focused, large enough to preserve context within a paragraph.
- chunk_overlap=50: chunks share a small overlapping window so that a
  sentence or idea split across a chunk boundary isn't completely lost
  from either chunk's context.
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent.parent / "data"


def load_documents() -> list[Document]:
    """Load every .txt file in the data directory as a LangChain Document,
    tagging each with its source filename as metadata. The metadata matters:
    when we retrieve a chunk later, we want to be able to tell the user
    which document it came from, not just show them raw text."""
    docs = []
    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks using a splitter that tries
    to break on paragraph boundaries first, then sentences, then words --
    in that priority order -- rather than cutting text at a fixed character
    count regardless of where it lands."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} source documents:")
    for d in docs:
        print(f"  - {d.metadata['source']} ({len(d.page_content)} chars)")

    chunks = split_documents(docs)
    print(f"\nSplit into {len(chunks)} chunks.")
    print("\nExample chunk:")
    print(f"  source: {chunks[0].metadata['source']}")
    print(f"  text: {chunks[0].page_content[:200]}...")
