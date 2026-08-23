"""
patch_generator_agent.py

Day 5: the third agent in the pipeline. Takes the bug localizer's top
suspect chunk and asks Qwen2.5-Coder to propose a fixed version of the
code, with an explanation of what changed and why.

Guardrail: local 7B models can generate code that looks plausible but is
not actually valid Python (mismatched brackets, bad indentation, etc.).
Before accepting a patch, it's validated with Python's own ast.parse —
if it doesn't parse, the patch is rejected and reported rather than
silently handed downstream as if it were usable.

This does NOT yet run tests against the patch (that's Day 6: sandboxed
test execution) or reflect on failures (Day 7). This agent's job is
narrowly: given a localized bug, propose one syntactically valid
candidate fix.

Usage:
    python src/patch_generator_agent.py "session cookies are not persisted across requests"
"""

from __future__ import annotations

import ast
import json
import sys

from langchain_ollama import ChatOllama

from bug_localizer_agent import BugLocalizerAgent
from planner_agent import PlannerAgent

MODEL_NAME = "qwen2.5-coder:7b"


class PatchGeneratorAgent:
    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)

    def generate_patch(self, issue_query: str, chunk: dict, extra_guidance: str | None = None) -> dict:
        if chunk["file_path"].startswith("tests/"):
            return {
                "chunk_id": chunk["id"],
                "status": "refused_test_file",
                "patched_code": None,
                "explanation": "Refusing to patch a test file — tests verify correctness and "
                                "should not be modified to make a patch appear to pass.",
            }

        prompt = self._build_prompt(issue_query, chunk, extra_guidance)
        response = self.llm.invoke(prompt)
        parsed = self._parse_response(response.content)

        if parsed is None:
            return {
                "chunk_id": chunk["id"],
                "status": "failed_to_parse_llm_response",
                "patched_code": None,
                "explanation": None,
            }

        patched_code = parsed.get("patched_code", "")
        is_valid, syntax_error = self._validate_syntax(patched_code)

        return {
            "chunk_id": chunk["id"],
            "status": "ok" if is_valid else "invalid_syntax",
            "original_code": chunk["code"],
            "patched_code": patched_code,
            "explanation": parsed.get("explanation", ""),
            "syntax_error": syntax_error,
        }

    # ---------- prompt + parsing ----------

    def _build_prompt(self, issue_query: str, chunk: dict, extra_guidance: str | None = None) -> str:
        guidance_block = f"\n\nIMPORTANT — a previous attempt failed. Guidance for this attempt:\n{extra_guidance}\n" if extra_guidance else ""
        return f"""You are fixing a bug in the Python "requests" HTTP client library.
{guidance_block}
Issue report: {issue_query}

The following function/method has been identified as the likely source of the bug:

id: {chunk['id']}
code:
{chunk['code']}

Propose a fixed version of this code. Preserve the function signature and overall
structure — only change what's needed to fix the described issue.

Respond in EXACTLY this format, nothing else before or after:

EXPLANATION: <one or two sentences on what changed and why>
PATCHED_CODE:
````python
<the full fixed function, complete and syntactically valid>
```"""

    @staticmethod
    def _parse_response(raw: str) -> dict | None:
        import re

        explanation_match = re.search(r"EXPLANATION:\s*(.+?)(?=PATCHED_CODE:|$)", raw, re.DOTALL)
        code_match = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)

        if not code_match:
            return None

        return {
            "explanation": explanation_match.group(1).strip() if explanation_match else "",
            "patched_code": code_match.group(1).strip(),
        }

    @staticmethod
    def _validate_syntax(code: str) -> tuple[bool, str | None]:
        if not code or not code.strip():
            return False, "empty patch"
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "session cookies are not persisted across requests"

    print("Running planner agent to gather context...")
    planner = PlannerAgent()
    plan_result = planner.run(query)
    print(f"Planner: {len(plan_result['context'])} chunks retrieved, sufficient={plan_result['sufficient']}\n")

    print("Running bug localization...")
    localizer = BugLocalizerAgent()
    loc_result = localizer.localize(query, plan_result["context"])

    if not loc_result["candidates"]:
        print("No suspect chunks identified — cannot generate a patch.")
        sys.exit(1)

    chunk_by_id = {c["id"]: c for c in plan_result["context"]}

    # walk candidates in ranked order, skip any that are test files
    top_candidate = None
    target_chunk = None
    for cand in loc_result["candidates"]:
        candidate_chunk = chunk_by_id.get(cand["chunk_id"])
        if candidate_chunk is None:
            continue
        if candidate_chunk["file_path"].startswith("tests/"):
            print(f"Skipping test-file candidate: {cand['chunk_id']}")
            continue
        top_candidate = cand
        target_chunk = candidate_chunk
        break

    if target_chunk is None:
        print("No non-test candidate found among suspects — cannot generate a patch.")
        sys.exit(1)

    print(f"Top suspect: {top_candidate['chunk_id']} [{top_candidate['confidence']}]\n")

    print("Generating patch...")
    patcher = PatchGeneratorAgent()
    patch_result = patcher.generate_patch(query, target_chunk)

    print(f"\nStatus: {patch_result['status']}")
    if patch_result["status"] == "ok":
        print(f"\nExplanation: {patch_result['explanation']}")
        print(f"\n--- Original ---\n{patch_result['original_code']}")
        print(f"\n--- Patched ---\n{patch_result['patched_code']}")
    else:
        print(f"Details: {patch_result.get('syntax_error', 'no valid patch produced')}")