"""
ast_parser.py

Parses a Python repository into function- and class-level chunks using
tree-sitter, and extracts the raw call/import names referenced inside
each chunk. This is the foundation for:
  1. The vector index (each chunk gets embedded)
  2. The dependency graph (chunk -> chunk edges, built in graph_builder.py
     by resolving the call names extracted here against chunk definitions)

Design note: this module deliberately does NOT resolve calls to their
definitions. It only *extracts* what each chunk calls/imports. Resolution
(turning "prepare_request" into a specific Chunk id) happens in
graph_builder.py, once we have the full symbol table across the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())


@dataclass
class Chunk:
    """A single function or class extracted from a repository."""

    id: str  # unique id: "relative/path.py::ClassName.func_name"
    file_path: str  # path relative to repo root
    name: str  # function or class name (qualified with class if a method)
    kind: str  # "function" | "method" | "class"
    code: str  # the raw source text of this chunk
    start_line: int
    end_line: int
    docstring: str | None = None
    calls: list[str] = field(default_factory=list)  # names this chunk calls
    imports: list[str] = field(default_factory=list)  # only populated for module-level chunk

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "name": self.name,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "docstring": self.docstring,
            "calls": self.calls,
            "imports": self.imports,
            "code": self.code,
        }


class RepoASTParser:
    """Walks a repository directory and extracts Chunks from every .py file."""

    SKIP_DIRS = {
        ".git", "__pycache__", ".venv", "venv", "env",
        "node_modules", "build", "dist", ".tox", ".eggs",
    }

    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root).resolve()
        self.parser = Parser(PY_LANGUAGE)

    # ---------- public API ----------

    def parse_repo(self) -> list[Chunk]:
        """Parse every .py file in the repo and return a flat list of Chunks."""
        chunks: list[Chunk] = []
        for py_file in self._iter_python_files():
            try:
                chunks.extend(self._parse_file(py_file))
            except (UnicodeDecodeError, SyntaxError) as e:
                print(f"  [skip] {py_file}: {e}")
        return chunks

    # ---------- internals ----------

    def _iter_python_files(self):
        for dirpath, dirnames, filenames in os.walk(self.repo_root):
            dirnames[:] = [d for d in dirnames if d not in self.SKIP_DIRS]
            for fname in filenames:
                if fname.endswith(".py"):
                    yield Path(dirpath) / fname

    def _parse_file(self, file_path: Path) -> list[Chunk]:
        source_bytes = file_path.read_bytes()
        tree = self.parser.parse(source_bytes)
        root = tree.root_node
        rel_path = str(file_path.relative_to(self.repo_root)).replace("\\", "/")

        chunks: list[Chunk] = []
        module_calls: list[str] = []
        module_imports: list[str] = []

        for node in root.children:
            if node.type == "function_definition":
                chunks.append(self._make_function_chunk(node, source_bytes, rel_path))
            elif node.type == "class_definition":
                chunks.append(self._make_class_chunk(node, source_bytes, rel_path))
                chunks.extend(self._make_method_chunks(node, source_bytes, rel_path))
            elif node.type in ("import_statement", "import_from_statement"):
                module_imports.extend(self._extract_import_names(node, source_bytes))
            else:
                # module-level statements (e.g. constants, top-level calls)
                module_calls.extend(self._extract_calls(node, source_bytes))

        if module_imports or module_calls:
            chunks.append(
                Chunk(
                    id=f"{rel_path}::<module>",
                    file_path=rel_path,
                    name="<module>",
                    kind="module",
                    code="",
                    start_line=1,
                    end_line=1,
                    calls=sorted(set(module_calls)),
                    imports=sorted(set(module_imports)),
                )
            )
        return chunks

    def _make_function_chunk(self, node, source_bytes, rel_path, class_name: str | None = None) -> Chunk:
        name_node = node.child_by_field_name("name")
        func_name = self._text(name_node, source_bytes)
        qualified = f"{class_name}.{func_name}" if class_name else func_name
        code = self._text(node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)
        calls = self._extract_calls(node, source_bytes)

        return Chunk(
            id=f"{rel_path}::{qualified}",
            file_path=rel_path,
            name=qualified,
            kind="method" if class_name else "function",
            code=code,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=docstring,
            calls=sorted(set(calls)),
        )

    def _make_class_chunk(self, node, source_bytes, rel_path) -> Chunk:
        name_node = node.child_by_field_name("name")
        class_name = self._text(name_node, source_bytes)
        code = self._text(node, source_bytes)
        docstring = self._extract_docstring(node, source_bytes)

        return Chunk(
            id=f"{rel_path}::{class_name}",
            file_path=rel_path,
            name=class_name,
            kind="class",
            code=code,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            docstring=docstring,
        )

    def _make_method_chunks(self, class_node, source_bytes, rel_path) -> list[Chunk]:
        name_node = class_node.child_by_field_name("name")
        class_name = self._text(name_node, source_bytes)
        body = class_node.child_by_field_name("body")
        methods = []
        if body is None:
            return methods
        for child in body.children:
            if child.type == "function_definition":
                methods.append(self._make_function_chunk(child, source_bytes, rel_path, class_name=class_name))
        return methods

    def _extract_docstring(self, node, source_bytes) -> str | None:
        body = node.child_by_field_name("body")
        if body is None or len(body.children) == 0:
            return None
        first_stmt = body.children[0]
        if first_stmt.type == "expression_statement" and first_stmt.children:
            expr = first_stmt.children[0]
            if expr.type == "string":
                text = self._text(expr, source_bytes)
                return text.strip("\"'").strip()
        return None

    def _extract_calls(self, node, source_bytes) -> list[str]:
        """Walk the subtree and collect the callee name of every call_expression."""
        calls = []

        def walk(n):
            if n.type == "call":
                func_node = n.child_by_field_name("function")
                if func_node is not None:
                    name = self._resolve_callee_name(func_node, source_bytes)
                    if name:
                        calls.append(name)
            for child in n.children:
                walk(child)

        walk(node)
        return calls

    def _resolve_callee_name(self, func_node, source_bytes) -> str | None:
        # Simple call: foo()
        if func_node.type == "identifier":
            return self._text(func_node, source_bytes)
        # Attribute call: self.foo(), module.foo(), obj.attr.foo()
        if func_node.type == "attribute":
            attr_node = func_node.child_by_field_name("attribute")
            if attr_node is not None:
                return self._text(attr_node, source_bytes)
        return None

    def _extract_import_names(self, node, source_bytes) -> list[str]:
        names = []
        for child in node.children:
            if child.type in ("dotted_name", "identifier"):
                names.append(self._text(child, source_bytes))
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    names.append(self._text(name_node, source_bytes))
        return names

    @staticmethod
    def _text(node, source_bytes) -> str:
        text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        return text.replace("\r\n", "\n").replace("\r", "\n")

if __name__ == "__main__":
    import json
    import sys

    repo_path = sys.argv[1] if len(sys.argv) > 1 else "data/test_repo"
    parser = RepoASTParser(repo_path)
    all_chunks = parser.parse_repo()

    print(f"Parsed {len(all_chunks)} chunks from {repo_path}\n")
    kinds = {}
    for c in all_chunks:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    print("By kind:", kinds)

    print("\nSample chunks:")
    for c in all_chunks[:5]:
        print(f"  [{c.kind}] {c.id}  (lines {c.start_line}-{c.end_line}, {len(c.calls)} calls)")

    out_path = "data/chunks.json"
    with open(out_path, "w") as f:
        json.dump([c.to_dict() for c in all_chunks], f, indent=2)
    print(f"\nSaved full chunk data to {out_path}")