"""
graph_builder.py

Day 2: builds on the Chunks produced by ast_parser.py.

Two responsibilities:
  1. Dependency graph: resolve the raw call names each chunk references
     (extracted by ast_parser.py) against a repo-wide symbol table, and
     build a NetworkX directed graph of chunk -> chunk "calls" edges.
  2. Embeddings: embed each chunk's code + docstring using CodeBERT and
     store the vectors in a persistent ChromaDB collection for semantic
     retrieval later (used by the hybrid retriever in later phases).

Scope: only chunks under src/ and tests/ are included (docs, setup.py,
etc. are excluded) — this keeps the graph focused on code that matters
for bug localization and patch generation.

Usage:
    python src/graph_builder.py [chunks_json_path]

Outputs:
    data/dependency_graph.json   (NetworkX node-link format)
    data/chroma_db/              (persistent ChromaDB collection)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

# ---------- scope filtering ----------

INCLUDE_PREFIXES = ("src/", "tests/")


def load_chunks(chunks_json_path: str) -> list[dict]:
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_in_scope(chunks: list[dict]) -> list[dict]:
    """Keep only chunks whose file_path starts with an included prefix."""
    return [c for c in chunks if c["file_path"].startswith(INCLUDE_PREFIXES)]


# ---------- symbol table + call resolution ----------

class SymbolResolver:
    """
    Maps raw call names (e.g. "prepare_request") to candidate chunk ids.

    Because ast_parser.py only extracts the *name* of a call, not its
    fully-qualified target, a name like "get" might match many chunks
    (methods named "get" across several classes). We resolve using a
    priority order:
      1. Exact match within the SAME FILE as the caller (most likely correct)
      2. If there's exactly one match anywhere in-scope, use it (unambiguous)
      3. Otherwise, skip the edge (too ambiguous to resolve safely)
    """

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        # name -> list of chunk ids that define something with that name
        self.by_name: dict[str, list[str]] = defaultdict(list)
        # (file_path, name) -> chunk id, for same-file lookups
        self.by_file_and_name: dict[tuple[str, str], str] = {}

        for c in chunks:
            if c["kind"] in ("function", "method", "class"):
                # index by simple name (methods: "ClassName.method" -> also index "method")
                simple_name = c["name"].split(".")[-1]
                self.by_name[simple_name].append(c["id"])
                self.by_file_and_name[(c["file_path"], simple_name)] = c["id"]

    def resolve(self, caller_file_path: str, called_name: str) -> str | None:
        # Priority 1: same file
        same_file_id = self.by_file_and_name.get((caller_file_path, called_name))
        if same_file_id is not None:
            return same_file_id

        # Priority 2: globally unique match
        candidates = self.by_name.get(called_name, [])
        if len(candidates) == 1:
            return candidates[0]

        # Ambiguous or unknown (e.g. stdlib call like "len", "isinstance") -> skip
        return None


def resolve_all_calls(chunks: list[dict]) -> tuple[list[tuple[str, str]], int, int]:
    """
    Returns (edges, resolved_count, unresolved_count).
    edges is a list of (caller_chunk_id, callee_chunk_id) tuples.
    """
    resolver = SymbolResolver(chunks)
    edges: list[tuple[str, str]] = []
    resolved, unresolved = 0, 0

    for c in chunks:
        for called_name in c.get("calls", []):
            target_id = resolver.resolve(c["file_path"], called_name)
            if target_id is not None and target_id != c["id"]:
                edges.append((c["id"], target_id))
                resolved += 1
            else:
                unresolved += 1

    return edges, resolved, unresolved


# ---------- graph construction ----------

def build_graph(chunks: list[dict], edges: list[tuple[str, str]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for c in chunks:
        G.add_node(
            c["id"],
            name=c["name"],
            kind=c["kind"],
            file_path=c["file_path"],
            start_line=c["start_line"],
            end_line=c["end_line"],
        )
    for caller, callee in edges:
        G.add_edge(caller, callee, relation="calls")
    return G


def save_graph(G: nx.DiGraph, out_path: str) -> None:
    data = nx.node_link_data(G, edges="edges")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------- embeddings (CodeBERT) ----------

class CodeEmbedder:
    """
    Wraps a retrieval-tuned sentence-transformers model.

    Note: raw CodeBERT (mean-pooled hidden states) was evaluated first, but
    produced poor retrieval quality — natural-language and code-style queries
    alike surfaced semantically unrelated chunks (e.g. exception classes for
    an HTTP request query). This is because CodeBERT is trained for code
    understanding tasks, not retrieval, so its embedding space isn't
    well-separated for nearest-neighbor search. all-MiniLM-L6-v2 is trained
    specifically for semantic similarity/retrieval and performs far better
    here, while also being smaller and faster.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        return vectors.tolist()


def chunk_text_for_embedding(c: dict) -> str:
    """What we actually embed: docstring (if any) + code, so semantic search
    can match on both natural-language intent and code structure."""
    parts = []
    if c.get("docstring"):
        parts.append(c["docstring"])
    parts.append(c.get("code", ""))
    return "\n".join(parts).strip()


def store_in_chromadb(chunks: list[dict], persist_dir: str, collection_name: str = "repo_chunks") -> None:
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    # Fresh collection each run, so re-running graph_builder.py doesn't duplicate/stale entries
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    embedder = CodeEmbedder()
    texts = [chunk_text_for_embedding(c) for c in chunks]
    print(f"Embedding {len(texts)} chunks with CodeBERT (first run downloads the model, ~500MB)...")
    embeddings = embedder.embed(texts)

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"file_path": c["file_path"], "kind": c["kind"], "name": c["name"]}
            for c in chunks
        ],
    )
    print(f"Stored {collection.count()} embeddings in ChromaDB at {persist_dir}")


# ---------- main ----------

if __name__ == "__main__":
    chunks_path = sys.argv[1] if len(sys.argv) > 1 else "data/chunks.json"

    all_chunks = load_chunks(chunks_path)
    in_scope = filter_in_scope(all_chunks)
    print(f"Loaded {len(all_chunks)} chunks total, {len(in_scope)} in scope (src/ + tests/)")

    edges, resolved, unresolved = resolve_all_calls(in_scope)
    print(f"Resolved {resolved} call edges, skipped {unresolved} (ambiguous/unknown/stdlib)")

    G = build_graph(in_scope, edges)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    graph_out = "data/dependency_graph.json"
    save_graph(G, graph_out)
    print(f"Saved dependency graph to {graph_out}")

    chroma_dir = "data/chroma_db"
    store_in_chromadb(in_scope, chroma_dir)