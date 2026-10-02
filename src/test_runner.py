
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

IMAGE_NAME = "repocoder-agent-sandbox"
REPO_ROOT = Path(__file__).resolve().parent.parent  # project root (one level up from src/)
BASE_REPO = REPO_ROOT / "data" / "test_repo"
DOCKERFILE = REPO_ROOT / "Dockerfile"


def build_image_if_needed() -> None:
    """Builds the sandbox image. Docker caches layers, so re-running this
    after the first build is fast unless requirements-dev.txt changed."""
    print(f"Building Docker image '{IMAGE_NAME}' (cached after first run)...")
    result = subprocess.run(
        ["docker", "build", "-t", IMAGE_NAME, "-f", str(DOCKERFILE), str(REPO_ROOT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed:\n{result.stdout}\n{result.stderr}")
    print("Image ready.")


def apply_patch(chunk: dict, patched_code: str, dest_repo: Path) -> None:
    """Replaces chunk.start_line..end_line in the chunk's file with patched_code,
    inside dest_repo (a fresh copy, not the original)."""
    target_file = dest_repo / chunk["file_path"]
    original_lines = target_file.read_text(encoding="utf-8").splitlines(keepends=True)

    start_idx = chunk["start_line"] - 1  # start_line is 1-indexed
    end_idx = chunk["end_line"]  # end_line is inclusive, so this slice is correct as-is

    # tree-sitter's node span starts at the first token (e.g. "def"), not at
    # the start of the line, so chunk["code"] is missing the original line's
    # leading indentation (relevant for methods inside classes). Recover it
    # from the line being replaced so the patch aligns with its enclosing block.
    original_first_line = original_lines[start_idx]
    leading_ws = original_first_line[: len(original_first_line) - len(original_first_line.lstrip())]

    # LLM-generated code starts at column 0 with body lines indented only
    # relative to that (unlike tree-sitter-extracted chunks, where every
    # line but the first already carries the file's absolute indentation).
    # So here we must prepend the file's leading indentation to EVERY
    # non-blank line, not just the first.
    patched_body_lines = patched_code.rstrip("\n").split("\n")
    patched_body_lines = [leading_ws + line if line.strip() else line for line in patched_body_lines]
    patched_text = "\n".join(patched_body_lines) + "\n"

    patched_lines = original_lines[:start_idx] + [patched_text] + original_lines[end_idx:]
    target_file.write_text("".join(patched_lines), encoding="utf-8")

def run_tests_on_patch(chunk: dict, patched_code: str, test_target: str, timeout_seconds: int = 120) -> dict:
    """
    chunk: the original chunk dict (needs file_path, start_line, end_line)
    patched_code: the proposed replacement code for that chunk
    test_target: relative path to the test file/dir to run, e.g. "tests/test_requests.py"

    Returns: {"passed": bool, "returncode": int, "stdout": str, "stderr": str, "timed_out": bool}
    """
    with tempfile.TemporaryDirectory(prefix="repocoder_patch_") as tmp:
        patched_repo = Path(tmp) / "repo"
        shutil.copytree(BASE_REPO, patched_repo)
        apply_patch(chunk, patched_code, patched_repo)

        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{patched_repo}:/repo",
                    IMAGE_NAME,
                    *(test_target if isinstance(test_target, list) else [test_target]),
                    "-v", "--tb=short",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return {
                "passed": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as e:
            return {
                "passed": False,
                "returncode": None,
                "stdout": (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""),
                "stderr": "Test run exceeded timeout — possible infinite loop introduced by patch.",
                "timed_out": True,
            }


if __name__ == "__main__":
    # quick manual test: run the unpatched repo's cookie tests, to sanity-check
    # the Docker plumbing works before wiring in real patches
    import json

    build_image_if_needed()

    chunks_path = REPO_ROOT / "data" / "chunks.json"
    with open(chunks_path, encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunk_by_id = {c["id"]: c for c in all_chunks}

    target_id = "src/requests/sessions.py::SessionRedirectMixin.resolve_redirects"
    target_chunk = chunk_by_id[target_id]

    print(f"Sanity check: running tests against UNPATCHED code for {target_id}")
    result = run_tests_on_patch(
        chunk=target_chunk,
        patched_code=target_chunk["code"],  # no actual change — just re-inserting original code
        test_target="tests/test_requests.py::TestRequests::test_request_cookies_not_persisted",
    )

    print(f"\nPassed: {result['passed']}")
    print(f"Return code: {result['returncode']}")
    print(f"Timed out: {result['timed_out']}")
    print(f"\n--- stdout ---\n{result['stdout']}")
    if result["stderr"]:
        print(f"\n--- stderr ---\n{result['stderr']}")