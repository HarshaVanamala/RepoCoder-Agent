# RepoCoder-Agent

A multi-agent AI system for automated bug localization and repair in GitHub repositories. Paste a bug report, and RepoCoder-Agent retrieves relevant code, localizes the likely fault, generates a patch, tests it in an isolated sandbox, and self-corrects on failure — all on local models, with no paid cloud API dependency.

Currently evaluated against [`psf/requests`](https://github.com/psf/requests), the Python HTTP library.

## What it does

1. Takes a natural-language bug report or GitHub-issue-style query.
2. Retrieves relevant code using both semantic search and structural (call-graph) context.
3. Localizes the most likely faulty function/method.
4. Generates a candidate patch and validates it syntactically.
5. Runs the real test suite against the patch inside a Docker sandbox.
6. If the patch fails, reflects on the failure and attempts one more informed retry.
7. Returns a tested, validated patch — or a clear explanation when no safe fix is found.

## Architecture / pipeline

```
Repository
   │
   ▼
AST Parser + Graph Builder
   │
   ▼
Hybrid Retriever (semantic + graph expansion)
   │
   ▼
Planner Agent (checks context sufficiency, reformulates if needed)
   │
   ▼
Bug Localizer Agent (ranks candidate fault locations)
   │
   ▼
Patch Generator Agent (proposes + syntax-validates a fix)
   │
   ▼
Docker Sandbox (runs real pytest suite)
   │
   ├── Pass → Output: tested, validated patch
   │
   └── Fail → Reflection Agent → guided retry (capped at 2 total attempts)
```

## Components

| File | Role |
|---|---|
| `src/ast_parser.py` | AST parsing and function/class-level chunk extraction |
| `src/graph_builder.py` | Builds a NetworkX call/import dependency graph over `src/` + `tests/`, saved to `data/dependency_graph.json` |
| `src/hybrid_retriever.py` | Combines ChromaDB/MiniLM semantic search with 1-hop graph expansion (callers + callees); shared by all downstream agents |
| `src/planner_agent.py` | LangGraph state machine: retrieve → assess sufficiency → reformulate (capped at 1 retry) → retrieve → done |
| `src/bug_localizer_agent.py` | Ranks top 1–3 candidate fault chunks by confidence via Qwen2.5-Coder; discards hallucinated chunk IDs |
| `src/patch_generator_agent.py` | Proposes a fix, validates syntax via `ast.parse`, refuses to patch test files or class-level chunks |
| `src/test_runner.py` + `Dockerfile` | Runs the real pytest suite in an isolated sandbox against a fresh temp copy per run |
| `src/reflection_agent.py` | Self-correction loop: generate → test → reflect → regenerate → test (capped at 2 total attempts) |
| `streamlit_app.py` | Interactive UI wrapping the full pipeline |
| `src/test_query.py` | Manual/debug script for spot-checking retrieval quality — not part of the core pipeline |

## Setup

**Requirements:**
- Python 3.10
- [Ollama](https://ollama.com) installed locally, with the model pulled:
  ```
  ollama pull qwen2.5-coder:7b
  ```
- Docker (with WSL2 backend on Windows) — required for sandboxed test execution

**Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the UI

```bash
streamlit run streamlit_app.py
```
Then describe a bug (e.g. *"session cookies are not persisted across requests"*), pick a test file/filter, and submit. The UI shows retrieval, localization, and each patch attempt live.

## How the Docker sandbox works

Each patch attempt is tested against a **fresh temporary copy** of the repository inside a Docker container — the original repo is never mutated. This guarantees that "passed tests" means the patch was actually validated by execution, not just syntax-checked or self-reported by the LLM.

## Example / demo flow

```bash
python src/reflection_agent.py "session cookies are not persisted across requests"
```
This runs the planner → localizer → patch generator → sandbox → reflection loop end-to-end from the command line and prints each attempt's result.

## Project structure

```
RepoCoder-Agent/
├── src/
│   ├── ast_parser.py
│   ├── graph_builder.py
│   ├── hybrid_retriever.py
│   ├── planner_agent.py
│   ├── bug_localizer_agent.py
│   ├── patch_generator_agent.py
│   ├── test_runner.py
│   ├── reflection_agent.py
│   ├── test_query.py          # manual retrieval debug script
│   └── test_bad_patch.py      # manual patch-testing debug script
├── data/                      # generated at runtime (gitignored)
├── Dockerfile
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Known limitations

- **Localization ranking:** in a controlled fault-injection test, retrieval correctly surfaced the buggy function, but the localizer ranked a caller function higher. Since only the top-ranked candidate is patched, the actual bug was missed. Candidate-fallback logic (retry lower-ranked candidates) is identified as the fix, scoped as future work.
- **Reflection grounding:** guidance generated between retry attempts is still fairly generic rather than grounded in the precise expected-vs-actual test failure detail.
- **Planner conservatism:** the local 7B model's context-sufficiency judgment is often overly conservative, flagging genuinely adequate context as insufficient. The pipeline hands off best-effort context regardless.
- **Single-language scope:** evaluated on Python repositories only; no multi-language support.