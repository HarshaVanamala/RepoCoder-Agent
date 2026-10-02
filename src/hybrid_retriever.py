
from __future__ import annotations

import json
from pathlib import Path

import chromadb
import networkx as nx

from graph_builder import CodeEmbedder


class HybridRetriever:
    def __init__(
        self,
        graph_path: str = "data/dependency_graph.json",
        chroma_path: str = "data/chroma_db",
        chunks_path: str = "data/chunks.json",
        collection_name: str = "repo_chunks",
    ):
        # dependency graph (built in Day 2)
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        self.graph: nx.DiGraph = nx.node_link_graph(graph_data, edges="edges")

        # full chunk data (code, docstring) — Chroma only stores the
        # embedded text + light metadata, so we look up the rest here
        with open(chunks_path, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
        self.chunks_by_id: dict[str, dict] = {c["id"]: c for c in all_chunks}

        # vector store (built in Day 2)
        client = chromadb.PersistentClient(path=chroma_path)
        self.collection = client.get_collection(collection_name)

        # must match the embedder used when the collection was built,
        # or Chroma raises a dimension-mismatch error (see memory notes)
        self.embedder = CodeEmbedder()

    def retrieve(
        self,
        query: str,
        top_k_semantic: int = 5,
        graph_hops: int = 1,
        max_results: int = 15,
    ) -> list[dict]:
        """
        Returns a ranked list of chunk dicts, each tagged with how it was
        found:
          "semantic_match" — directly matched the query's meaning
          "calls"          — called by a semantic match (callee context)
          "called_by"      — calls a semantic match (caller context)
        Semantic matches always rank first, since they're the strongest
        signal; graph-expanded chunks fill in structural context after.
        """
        seed_ids = self._semantic_search(query, top_k_semantic)

        results: list[dict] = []
        seen: set[str] = set()

        for rank, chunk_id in enumerate(seed_ids):
            if chunk_id in seen or chunk_id not in self.chunks_by_id:
                continue
            results.append(self._make_result(chunk_id, "semantic_match", rank))
            seen.add(chunk_id)

        for chunk_id in list(seed_ids):
            for neighbor_id, relation in self._expand_graph(chunk_id, graph_hops):
                if neighbor_id in seen or neighbor_id not in self.chunks_by_id:
                    continue
                results.append(self._make_result(neighbor_id, relation, None))
                seen.add(neighbor_id)

        return results[:max_results]

    # ---------- internals ----------

    def _semantic_search(self, query: str, top_k: int) -> list[str]:
        query_vec = self.embedder.embed([query])[0]
        r = self.collection.query(query_embeddings=[query_vec], n_results=top_k)
        return r["ids"][0]

    def _expand_graph(self, chunk_id: str, hops: int) -> list[tuple[str, str]]:
        """Returns (neighbor_id, relation) pairs within `hops` steps."""
        if chunk_id not in self.graph:
            return []

        expanded: list[tuple[str, str]] = []
        # callees: functions this chunk calls
        for callee_id in nx.descendants_at_distance(self.graph, chunk_id, 1) if hops >= 1 else []:
            expanded.append((callee_id, "calls"))
        # callers: functions that call this chunk (reverse direction)
        reverse = self.graph.reverse(copy=False)
        for caller_id in nx.descendants_at_distance(reverse, chunk_id, 1) if hops >= 1 else []:
            expanded.append((caller_id, "called_by"))

        return expanded

    def _make_result(self, chunk_id: str, relation: str, semantic_rank: int | None) -> dict:
        c = self.chunks_by_id[chunk_id]
        return {
            "id": c["id"],
            "kind": c["kind"],
            "name": c["name"],
            "file_path": c["file_path"],
            "code": c["code"],
            "docstring": c.get("docstring"),
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            "relation": relation,
            "semantic_rank": semantic_rank,
        }


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "how does the library handle redirects?"
    retriever = HybridRetriever()
    results = retriever.retrieve(query)

    print(f"Query: {query!r}\n")
    for r in results:
        tag = r["relation"]
        rank = f"(rank {r['semantic_rank']})" if r["semantic_rank"] is not None else ""
        print(f"  [{tag:14s}] {r['kind']:8s} {r['id']} {rank}")