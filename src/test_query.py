import sys
sys.path.insert(0, str(Path(__file__).parent))
from graph_builder import CodeEmbedder
import chromadb

embedder = CodeEmbedder()

queries = [
    "send an HTTP GET request",        # natural language
    "def get(self, url",               # code-style snippet
    "requests.get",                    # API-call style
    "raise an exception for HTTP errors",  # natural language, different topic
]

c = chromadb.PersistentClient(path="data/chroma_db")
col = c.get_collection("repo_chunks")

for q in queries:
    vec = embedder.embed([q])[0]
    r = col.query(query_embeddings=[vec], n_results=3)
    print(f"\nQuery: {q!r}")
    for id_, meta in zip(r["ids"][0], r["metadatas"][0]):
        print(f"  {meta['kind']:8s} {id_}")