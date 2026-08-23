# RepoCoder-Agent — Day 1

An AST-aware, explainable code-understanding assistant for open-source
contributors. Day 1 builds the foundation everything else sits on: a
tree-sitter parser that turns a Python repository into structured chunks
with call/import references extracted.

## What's here

```
repocoder-agent/
├── src/
│   └── ast_parser.py       # RepoASTParser: repo -> list[Chunk]
├── data/
│   ├── test_repo/          # psf/requests, cloned as the working example
│   └── chunks.json         # output of running the parser on test_repo
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run it

```bash
python src/ast_parser.py data/test_repo
```

This walks every `.py` file in the repo and produces a `Chunk` for each
function, method, and class, plus one synthetic `<module>` chunk per file
capturing module-level imports and top-level calls. Output is written to
`data/chunks.json`.

## What a Chunk looks like

```json
{
  "id": "src/requests/sessions.py::Session.send",
  "file_path": "src/requests/sessions.py",
  "name": "Session.send",
  "kind": "method",
  "start_line": 752,
  "end_line": 829,
  "docstring": "Send a given PreparedRequest.",
  "calls": ["get_adapter", "resolve_redirects", "dispatch_hook", "send", "..."],
  "imports": [],
  "code": "def send(self, request, **kwargs): ..."
}
```

The `calls` list is the raw names referenced inside the chunk — **not yet
resolved** to specific definitions. That resolution (turning the string
`"get_adapter"` into a pointer at `src/requests/sessions.py::Session.get_adapter`)
is deliberately deferred to `graph_builder.py` (Day 2), which needs the
full cross-file symbol table before it can disambiguate names that appear
in multiple classes/files.

## Design decisions worth knowing for the paper

- **Chunking granularity**: function/method/class, not statement-level
  (unlike GraphCoder's control/data-flow graph) and not whole-file
  (unlike naive RAG baselines). This is a deliberate middle ground: fine
  enough for precise localization, coarse enough to stay embeddable as a
  single unit and cheap to retrieve.
- **Call extraction is name-based, not type-resolved**: `self.foo()`,
  `module.foo()`, and a free-standing `foo()` all just record `"foo"`.
  This is intentional for Day 1 — full type resolution (know that
  `self.foo` really means `Session.foo`) needs either static type
  inference or the class context, which `graph_builder.py` adds next by
  matching call names against known chunk names scoped by file/class
  first, falling back to global name search.
- **One module-level chunk per file**: captures file-level imports and
  top-level statements (decorators registries, constants) without
  polluting per-function chunk boundaries.

## Verified against a real repo

Ran against `psf/requests` (37 files): 603 chunks — 104 functions, 70
classes, 394 methods, 35 module-level chunks. Spot-checked
`Session.send` and confirmed it correctly extracts calls into
`adapters.py` (`get_adapter`) and back into `sessions.py`
(`resolve_redirects`) — the actual cross-file dependency chain your
abstract's graph-expansion example describes.

## Next (Day 2)

- `graph_builder.py`: resolve `calls` names into a NetworkX `DiGraph` of
  chunk-to-chunk edges (this is your dependency graph).
- Embed each chunk's `code` + `docstring` with a code embedding model and
  store in ChromaDB.
- Note: `sentence-transformers`/`chromadb` need a Hugging Face model
  download (`microsoft/codebert-base`, ~400MB) — do this on your own
  machine, not in a network-restricted sandbox, the first time.