"""
chunking.py
===========

WHAT THIS FILE DOES (in plain words):
    We take Python source files and cut them into small, searchable pieces
    called "chunks". Each chunk = one function (or one method inside a class),
    plus a bit of metadata about it (its name, which file it's in, what it
    imports, what it calls, and its docstring/leading comment).

WHY FUNCTION-LEVEL, NOT FILE-LEVEL?
    A whole file is usually too big and covers too many unrelated things.
    A single function is small enough to be "about one thing", which is
    exactly what a retrieval query is usually asking about.

WHY METADATA, NOT JUST RAW CODE?
    Natural-language queries use words like "preprocess", "validate",
    "connect to database". Raw code often doesn't contain those exact words
    -- but docstrings, comments, and function names often do. Attaching
    that text to the chunk closes the "vocabulary gap" between how humans
    ask questions and how code is actually written.

This file has ZERO third-party dependencies -- only Python's built-in `ast`
module, which parses Python source into a tree we can walk over. This
keeps it fast, dependency-free, and easy to explain line-by-line.
"""

import ast
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CodeChunk:
    """
    One retrievable unit: a single function (or method).

    Every field here maps directly to a row in the "structured
    representation" table from the strategy doc (section 11):

        Function: <name>
        File: <path>
        Class: <parent class, if any>
        Imports: <...>
        Called functions: <...>
        Docstring/Comments: <...>
        Code: <raw source>
    """
    chunk_id: str            # stable ID: hash of file_path + function_name
    function_name: str
    file_path: str
    class_name: Optional[str]
    imports: List[str]
    called_functions: List[str]
    docstring: str
    code: str                # the raw source code of the function
    content_hash: str = field(default="")  # hash of just the code body

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = _hash_text(self.code)

    def to_indexable_text(self) -> str:
        """
        Turns this chunk into ONE string of text that we feed into both
        BM25 and the embedding model. This is the "structured representation"
        from the strategy doc: metadata first (because it usually contains
        the human-friendly words), then the raw code.
        """
        parts = [
            f"Function: {self.function_name}",
            f"File: {self.file_path}",
        ]
        if self.class_name:
            parts.append(f"Class: {self.class_name}")
        if self.imports:
            parts.append(f"Imports: {', '.join(self.imports)}")
        if self.called_functions:
            parts.append(f"Called functions: {', '.join(self.called_functions)}")
        if self.docstring:
            parts.append(f"Docstring: {self.docstring}")
        parts.append(f"Code:\n{self.code}")
        return "\n".join(parts)


def _hash_text(text: str) -> str:
    """Short, stable hash used as an ID / change-detector. See versioning.py."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _get_called_function_names(func_node: ast.FunctionDef) -> List[str]:
    """
    Walk the function body and collect the names of every function it calls,
    e.g. `x = normalize(s)` -> "normalize".
    We keep this simple on purpose: we only grab the *name* being called,
    not a full call graph. That's enough signal for retrieval.
    """
    called = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    return sorted(called)


def _get_module_imports(tree: ast.Module) -> List[str]:
    """Collect every top-level `import x` / `from x import y` in the file."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.extend(f"{module}.{alias.name}" for alias in node.names)
    return imports


def _trim_to_max_tokens(code: str, max_chars: int = 4000) -> str:
    """
    Guard against a chunk being absurdly long (strategy doc section 11:
    "avoid chunks becoming too large"). We keep the START (signature +
    first lines, usually the setup logic) and the END (often the return
    statement / final API calls) rather than blindly cutting off the tail.
    """
    if len(code) <= max_chars:
        return code
    head = code[: max_chars // 2]
    tail = code[-max_chars // 2:]
    return head + "\n# ... (truncated) ...\n" + tail


def chunk_python_file(file_path: str, source_code: str) -> List[CodeChunk]:
    """
    Main entry point: parse one Python file's source code and return a list
    of CodeChunk objects, one per function/method found in it.

    If the file has a syntax error (e.g. it's not valid Python, or it's
    Python 2), we just skip it and return an empty list rather than
    crashing the whole indexing job.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    module_imports = _get_module_imports(tree)
    chunks: List[CodeChunk] = []

    class _Visitor(ast.NodeVisitor):
        def __init__(self):
            self.current_class = None

        def visit_ClassDef(self, node: ast.ClassDef):
            previous_class = self.current_class
            self.current_class = node.name
            self.generic_visit(node)
            self.current_class = previous_class

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._handle_function(node)
            # do NOT recurse into nested functions separately for this
            # simple version -- keeps chunk boundaries clean and easy to
            # reason about.

        visit_AsyncFunctionDef = visit_FunctionDef

        def _handle_function(self, node):
            code_segment = ast.get_source_segment(source_code, node) or ""
            docstring = ast.get_docstring(node) or ""
            chunk = CodeChunk(
                chunk_id=_hash_text(f"{file_path}::{self.current_class or ''}::{node.name}"),
                function_name=node.name,
                file_path=file_path,
                class_name=self.current_class,
                imports=module_imports,
                called_functions=_get_called_function_names(node),
                docstring=docstring,
                code=_trim_to_max_tokens(code_segment),
            )
            chunks.append(chunk)

    _Visitor().visit(tree)
    return chunks


def chunk_repository(file_path_to_source: dict) -> List[CodeChunk]:
    """
    Convenience wrapper: chunk many files at once.
    `file_path_to_source` is just {file_path: source_code_string}.
    """
    all_chunks: List[CodeChunk] = []
    for path, source in file_path_to_source.items():
        all_chunks.extend(chunk_python_file(path, source))
    return all_chunks
