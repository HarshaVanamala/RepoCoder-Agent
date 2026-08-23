"""
reflection_agent.py

Day 7: closes the self-correction loop. When a generated patch fails its
test run (test_runner.py), this module:
  1. Extracts the meaningful failure signal from raw pytest output (error
     type, message, the specific line that failed) rather than dumping
     the whole verbose traceback at the LLM — keeps the prompt focused
     and fast on CPU.
  2. Asks Qwen2.5-Coder to reflect: WHY did this patch fail, and what
     should change in a retry.
  3. Feeds that reflection back into patch_generator_agent.py as extra
     context for ONE regeneration attempt.

Design: capped at 2 total patch attempts (1 initial + 1 reflected retry).
This bounds runtime on CPU, avoids the agent looping on similar bad
fixes indefinitely, and is enough to demonstrate a genuine reflection-
based self-correction loop — the core novelty claimed in the abstract.

Usage:
    python src/reflection_agent.py "session cookies are not persisted across requests"
"""

from __future__ import annotations

import re
import sys

from langchain_ollama import ChatOllama

from bug_localizer_agent import BugLocalizerAgent
from patch_generator_agent import PatchGeneratorAgent
from planner_agent import PlannerAgent
from test_runner import build_image_if_needed, run_tests_on_patch

MODEL_NAME = "qwen2.5-coder:7b"
MAX_ATTEMPTS = 2


class ReflectionAgent:
    def __init__(self):
        self.llm = ChatOllama(model=MODEL_NAME, temperature=0)

    def reflect(self, issue_query: str, chunk: dict, patched_code: str, test_result: dict) -> str:
        """Returns a short piece of guidance text to feed into the next
        patch generation attempt."""
        failure_signal = self._extract_failure_signal(test_result)

        prompt = f"""A proposed patch to fix a bug in the Python "requests" HTTP client library
FAILED its tests. Analyze why and give guidance for a better fix.

Issue report: {issue_query}

The patch that was tried:
{patched_code}

Test failure:
{failure_signal}

Respond with 2-3 sentences of specific, actionable guidance for what the next patch attempt
should do differently. Focus on the ROOT CAUSE shown in the failure, not generic advice."""

        response = self.llm.invoke(prompt)
        return response.content.strip()

    @staticmethod
    def _extract_failure_signal(test_result: dict) -> str:
        """Pulls out the error type + message + failing source line from raw
        pytest output, instead of passing the entire verbose traceback."""
        stdout = test_result.get("stdout", "")

        if test_result.get("timed_out"):
            return "The test run timed out — the patch likely introduced an infinite loop or hang."

        # pytest error lines look like: "E   NameError: name 'headers' is not defined"
        error_lines = re.findall(r"^E\s+(.+)$", stdout, re.MULTILINE)
        # the specific source line pytest points to, e.g. "src/requests/sessions.py:276: in resolve_redirects"
        location_lines = re.findall(r"^(\S+\.py:\d+): in (\S+)$", stdout, re.MULTILINE)

        summary_match = re.search(r"={3,}\s*short test summary info\s*={3,}\n(.+?)(?:\n={3,}|\Z)", stdout, re.DOTALL)

        parts = []
        if summary_match:
            parts.append(f"Failed tests:\n{summary_match.group(1).strip()}")
        if location_lines:
            loc, func = location_lines[-1]
            parts.append(f"Failure occurred at: {loc} (in {func})")
        if error_lines:
            parts.append("Error(s):\n" + "\n".join(error_lines[:5]))

        if not parts:
            # fall back to a trimmed raw excerpt if the format didn't match expectations
            return stdout[-1500:] if stdout else "(no output captured)"

        return "\n\n".join(parts)


def run_patch_with_reflection(issue_query: str, chunk: dict, test_target) -> dict:
    """
    Full attempt-loop: generate -> test -> (if fail) reflect -> regenerate -> test.
    Returns a dict describing the final outcome across all attempts.
    """
    patcher = PatchGeneratorAgent()
    reflector = ReflectionAgent()

    guidance = None
    attempts_log = []

    for attempt_num in range(1, MAX_ATTEMPTS + 1):
        gen_result = patcher.generate_patch(issue_query, chunk, extra_guidance=guidance)

        if gen_result["status"] != "ok":
            attempts_log.append({"attempt": attempt_num, "stage": "generation_failed", "detail": gen_result})
            continue

        test_result = run_tests_on_patch(chunk, gen_result["patched_code"], test_target)
        attempts_log.append({
            "attempt": attempt_num,
            "stage": "tested",
            "patched_code": gen_result["patched_code"],
            "explanation": gen_result["explanation"],
            "test_passed": test_result["passed"],
            "test_result": test_result,
        })

        if test_result["passed"]:
            return {"final_status": "passed", "attempts": attempts_log, "attempt_count": attempt_num}

        if attempt_num < MAX_ATTEMPTS:
            guidance = reflector.reflect(issue_query, chunk, gen_result["patched_code"], test_result)
            attempts_log[-1]["reflection"] = guidance

    return {"final_status": "failed_after_max_attempts", "attempts": attempts_log, "attempt_count": MAX_ATTEMPTS}


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "session cookies are not persisted across requests"

    print("Running planner + localizer...")
    planner = PlannerAgent()
    plan_result = planner.run(query)

    localizer = BugLocalizerAgent()
    loc_result = localizer.localize(query, plan_result["context"])
    print(f"Localizer candidates: {[(c['chunk_id'], c.get('confidence')) for c in loc_result['candidates']]}")
    print(f"Discarded hallucinations: {loc_result['discarded_hallucinations']}")

    chunk_by_id = {c["id"]: c for c in plan_result["context"]}
    target_chunk = None
    for cand in loc_result["candidates"]:
        c = chunk_by_id.get(cand["chunk_id"])
        if c and not c["file_path"].startswith("tests/"):
            target_chunk = c
            break

    if target_chunk is None:
        print("No valid non-test candidate found — aborting.")
        sys.exit(1)

    print(f"Target: {target_chunk['id']}\n")

    build_image_if_needed()

    test_keyword = sys.argv[2] if len(sys.argv) > 2 else "cookie or redirect"
    test_file = sys.argv[3] if len(sys.argv) > 3 else "tests/test_requests.py"
    result = run_patch_with_reflection(
        query, target_chunk, test_target=[test_file, "-k", test_keyword]
    )

    print(f"\nFinal status: {result['final_status']} (after {result['attempt_count']} attempt(s))\n")
    for a in result["attempts"]:
        print(f"--- Attempt {a['attempt']} ---")
        if a["stage"] == "generation_failed":
            print(f"  Generation failed: {a['detail']}")
            continue
        print(f"  Explanation: {a['explanation']}")
        print(f"  Test passed: {a['test_passed']}")
        if not a["test_passed"]:
            print(f"  Test output:\n{a['test_result']['stdout'][-1500:]}")
            if a["test_result"].get("stderr"):
                print(f"  Test stderr:\n{a['test_result']['stderr'][-500:]}")
        if "reflection" in a:
            print(f"  Reflection guidance for next attempt: {a['reflection']}")
        print()