import json
from pathlib import Path
from test_runner import run_tests_on_patch, build_image_if_needed

REPO_ROOT = Path(__file__).resolve().parent.parent
chunks_path = REPO_ROOT / "data" / "chunks.json"
with open(chunks_path, encoding="utf-8") as f:
    all_chunks = json.load(f)
chunk_by_id = {c["id"]: c for c in all_chunks}

target_id = "src/requests/sessions.py::SessionRedirectMixin.resolve_redirects"
target_chunk = chunk_by_id[target_id]

# the model's earlier "patch" that removed:  headers.pop("Cookie", None)
bad_patched_code = target_chunk["code"].replace(
    '            headers = prepared_request.headers\n            headers.pop("Cookie", None)\n\n',
    ""
)

build_image_if_needed()

print("Running FULL cookie/redirect test suite against the BAD patch...")
result = run_tests_on_patch(
    chunk=target_chunk,
    patched_code=bad_patched_code,
    test_target=["tests/test_requests.py", "-k", "cookie or redirect"],
)

print(f"\nPassed: {result['passed']}")
print(f"Return code: {result['returncode']}")
print(f"\n--- stdout ---\n{result['stdout']}")
if result["stderr"]:
    print(f"\n--- stderr ---\n{result['stderr']}")