"""
bug_localizer_agent.py

Day 4 (continued): the second agent in the pipeline. Takes the context
bundle produced by planner_agent.py and asks Qwen2.5-Coder to identify
which specific chunk(s) are most likely responsible for the described
bug, ranked with confidence and reasoning.

Guardrail: local 7B models can hallucinate plausible-sounding but
nonexistent identifiers. This agent validates every chunk_id the LLM
returns against the actual context it was given — any id that isn't in
the provided context is discarded rather than trusted, and logged as
such so the failure is visible rather than silent.

Usage:
    python src/bug_localizer_agent.py "session cookies are not persisted across requests"
"""

from __future__ import annotations

import json
import sys

from langchain_ollama import ChatOllama

from planner_agent import PlannerAgent

MODEL_NAME = "qwen2.5-coder:7b"
MAX_CODE_CHARS = 800  # truncate per-chunk code shown to the LLM, keeps prompt size sane on CPU


class BugLocalizerAgent:
    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)

    def localize(self, issue_query: str, context: list[dict]) -> dict:
        valid_ids = {c["id"] for c in context}
        prompt = self._build_prompt(issue_query, context)
        response = self.llm.invoke(prompt)
        candidates = self._parse_candidates(response.content)

        accepted, discarded = [], []
        for cand in candidates:
            if cand.get("chunk_id") in valid_ids:
                accepted.append(cand)
            else:
                discarded.append(cand)

        return {
            "issue": issue_query,
            "candidates": accepted,
            "discarded_hallucinations": discarded,
        }

    # ---------- prompt + parsing ----------

    def _build_prompt(self, issue_query: str, context: list[dict]) -> str:
        chunk_listing = []
        for c in context:
            code_snippet = c["code"][:MAX_CODE_CHARS]
            chunk_listing.append(
                f"id: {c['id']}\nkind: {c['kind']}\ncode:\n{code_snippet}\n---"
            )
        chunks_text = "\n".join(chunk_listing)

        return f"""You are localizing a bug in the Python "requests" HTTP client library.

Issue report: {issue_query}

Below are candidate code chunks retrieved from the repository. Identify the 1-3 chunks
MOST LIKELY responsible for this issue, ranked by likelihood. You MUST only use the exact
"id" values shown below — do not invent or modify ids.

Candidate chunks:
{chunks_text}

Respond ONLY with valid JSON, no other text, in this exact format:
{{"candidates": [{{"chunk_id": "<exact id from above>", "confidence": "high" or "medium" or "low", "reasoning": "one sentence"}}]}}"""

    @staticmethod
    def _parse_candidates(raw: str) -> list[dict]:
        try:
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(cleaned)
            return data.get("candidates", [])
        except (json.JSONDecodeError, AttributeError):
            return []


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "session cookies are not persisted across requests"

    print("Running planner agent to gather context...")
    planner = PlannerAgent()
    plan_result = planner.run(query)
    print(f"Planner: {len(plan_result['context'])} chunks retrieved, sufficient={plan_result['sufficient']}\n")

    print("Running bug localization...")
    localizer = BugLocalizerAgent()
    result = localizer.localize(query, plan_result["context"])

    print(f"\nIssue: {result['issue']}")
    print(f"\nTop suspect chunks:")
    for i, cand in enumerate(result["candidates"], 1):
        print(f"  {i}. [{cand.get('confidence', '?')}] {cand.get('chunk_id')}")
        print(f"     reason: {cand.get('reasoning')}")

    if result["discarded_hallucinations"]:
        print(f"\nDiscarded (hallucinated ids not in context): {len(result['discarded_hallucinations'])}")
        for d in result["discarded_hallucinations"]:
            print(f"  - {d.get('chunk_id')}")