"""
planner_agent.py

Day 4: the first agent in the multi-agent pipeline. Given a user query or
GitHub issue, it:
  1. Retrieves context using the Day 3 HybridRetriever
  2. Asks Qwen2.5-Coder (via Ollama) whether the retrieved context is
     SUFFICIENT to localize/understand the issue
  3. If not sufficient, asks Qwen2.5-Coder to REFORMULATE the query into
     more code-like/specific terms (bug reports use vague language; code
     uses precise names — this bridges that gap) and retrieves again
  4. Caps at one reformulation attempt (2 retrieval rounds total) to keep
     runtime bounded on CPU — this is a deliberate, documented enhancement
     over the original abstract's simpler "verify adequacy" planner.

The final state (context + sufficiency verdict) is what gets handed to
the next agent (bug localizer) in later phases.

Usage:
    python src/planner_agent.py "session cookies are not persisted across requests"
"""

from __future__ import annotations

import json
import sys
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from hybrid_retriever import HybridRetriever

MODEL_NAME = "qwen2.5-coder:7b"
MAX_ATTEMPTS = 2  # 1 initial retrieval + 1 reformulated retry


class PlannerState(TypedDict):
    original_query: str
    current_query: str
    attempt: int
    context: list[dict]
    sufficient: bool
    reasoning: str


class PlannerAgent:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)
        self.graph = self._build_graph()

    def run(self, query: str) -> PlannerState:
        initial_state: PlannerState = {
            "original_query": query,
            "current_query": query,
            "attempt": 0,
            "context": [],
            "sufficient": False,
            "reasoning": "",
        }
        return self.graph.invoke(initial_state)

    # ---------- graph construction ----------

    def _build_graph(self):
        g = StateGraph(PlannerState)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("assess", self._assess_node)
        g.add_node("reformulate", self._reformulate_node)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "assess")
        g.add_conditional_edges(
            "assess",
            self._route_after_assess,
            {"reformulate": "reformulate", "done": END},
        )
        g.add_edge("reformulate", "retrieve")

        return g.compile()

    # ---------- nodes ----------

    def _retrieve_node(self, state: PlannerState) -> dict:
        state["attempt"] += 1
        results = self.retriever.retrieve(state["current_query"])
        return {"context": results, "attempt": state["attempt"]}

    def _assess_node(self, state: PlannerState) -> dict:
        summary = self._summarize_context(state["context"])
        prompt = f"""You are judging whether retrieved code context is sufficient to investigate this issue.

Issue: {state['original_query']}

Retrieved code chunks:
{summary}

Respond ONLY with valid JSON, no other text, in this exact format:
{{"sufficient": true or false, "reasoning": "one sentence why"}}"""

        response = self.llm.invoke(prompt)
        verdict = self._parse_verdict(response.content)
        return {"sufficient": verdict["sufficient"], "reasoning": verdict["reasoning"]}

    def _reformulate_node(self, state: PlannerState) -> dict:
        prompt = f"""The following search query did not retrieve sufficient code context for this issue,
in the codebase for the Python "requests" HTTP client library (not Django, not any web framework —
this is a library for making HTTP requests).

Original issue: {state['original_query']}
Search query tried: {state['current_query']}
Why it fell short: {state['reasoning']}

Rewrite the search query using specific terminology that would plausibly appear in THIS codebase \
(e.g. Session, cookiejar, PreparedRequest, adapters, redirects — real concepts from the requests library). \
Do not introduce unrelated frameworks or technologies. Respond with ONLY the rewritten query text, nothing else."""

        response = self.llm.invoke(prompt)
        new_query = response.content.strip().strip('"')
        return {"current_query": new_query}

    # ---------- routing ----------

    def _route_after_assess(self, state: PlannerState) -> str:
        if state["sufficient"]:
            return "done"
        if state["attempt"] >= MAX_ATTEMPTS:
            return "done"  # give up gracefully, hand off best-effort context
        return "reformulate"

    # ---------- helpers ----------

    @staticmethod
    def _summarize_context(context: list[dict]) -> str:
        if not context:
            return "(no chunks retrieved)"
        lines = []
        for c in context[:10]:  # cap prompt size
            doc = f" - {c['docstring']}" if c.get("docstring") else ""
            lines.append(f"[{c['kind']}] {c['id']}{doc}")
        return "\n".join(lines)

    @staticmethod
    def _parse_verdict(raw: str) -> dict:
        try:
            # models sometimes wrap JSON in ```json fences despite instructions
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(cleaned)
            return {"sufficient": bool(data.get("sufficient", False)), "reasoning": str(data.get("reasoning", ""))}
        except (json.JSONDecodeError, AttributeError):
            # fail safe: if we can't parse the verdict, treat as insufficient
            # so the pipeline retries rather than silently proceeding on bad data
            return {"sufficient": False, "reasoning": f"Could not parse LLM verdict: {raw[:100]}"}


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "session cookies are not persisted across requests"

    agent = PlannerAgent()
    result = agent.run(query)

    print(f"Original query: {result['original_query']!r}")
    print(f"Final query used: {result['current_query']!r}")
    print(f"Attempts: {result['attempt']}")
    print(f"Sufficient: {result['sufficient']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"\nFinal context ({len(result['context'])} chunks):")
    for c in result["context"]:
        print(f"  [{c['relation']:14s}] {c['kind']:8s} {c['id']}")