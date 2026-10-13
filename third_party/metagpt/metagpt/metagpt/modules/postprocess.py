#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Post-processing Layer: automatic fixes after code generation.

(A) Import completion
(B) Function signature enforcement (entry_point wrapper)
(C) Basic syntax safety (markdown strip, code extraction)
"""
from __future__ import annotations

import ast
import re
from typing import Optional


# --- (C) Basic Syntax Safety ---


def strip_markdown_blocks(text: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` wrappers."""
    text = (text or "").strip()
    for pattern in [
        r"^```(?:python|py)\s*\n?(.*?)\n?```\s*$",
        r"^```\s*\n?(.*?)\n?```\s*$",
    ]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return text


def extract_code_block(text: str) -> str:
    """Extract the main code block, strip extra text before/after.

    Heuristics:
    - Remove markdown fences first.
    - If extra prose exists, keep only from the first real code line:
      an import/from/def/class statement.
    """
    text = strip_markdown_blocks(text)
    if not text:
        return text

    lines = text.splitlines()
    start_idx = None
    code_start = re.compile(r"^\s*(?:from\s+\w+|import\s+\w+|def\s+\w+\s*\(|class\s+\w+\s*[:\(])")
    for i, line in enumerate(lines):
        if code_start.search(line):
            start_idx = i
            break
    if start_idx is None:
        return text.strip()

    extracted = "\n".join(lines[start_idx:]).strip()
    return extracted or text.strip()


def _strip_top_level_asserts_and_main(code: str) -> str:
    """Remove common non-solution tail content.

    HumanEval-style evaluators execute the whole module; top-level asserts or
    ad-hoc "Test cases" blocks can cause false failures.
    """
    if not (code or "").strip():
        return code

    lines = code.splitlines()
    kept: list[str] = []
    skip_rest = False

    main_guard = re.compile(r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*$')
    for line in lines:
        if skip_rest:
            continue
        if main_guard.match(line):
            skip_rest = True
            continue

        # Drop top-level asserts (including those preceded by whitespace).
        if re.match(r"^\s*assert\b", line):
            continue
        kept.append(line)

    return "\n".join(kept).rstrip()


def _dedupe_entry_point_defs(code: str, entry_point: Optional[str]) -> str:
    """If entry_point is defined multiple times, keep the best one.

    Heuristic: choose the definition block with the most non-empty, non-comment
    lines (usually the real implementation), drop other duplicates.
    """
    ep = (entry_point or "").strip()
    if not ep:
        return code

    # Find all top-level def entry_point occurrences.
    pat = re.compile(rf"^def\s+{re.escape(ep)}\s*\(", re.MULTILINE)
    matches = list(pat.finditer(code))
    if len(matches) <= 1:
        return code

    lines = code.splitlines()
    # Map from absolute char index to line index
    char_to_line = []
    acc = 0
    for i, ln in enumerate(lines):
        char_to_line.append(acc)
        acc += len(ln) + 1

    def char_idx_to_line_idx(cidx: int) -> int:
        # last line whose start <= cidx
        lo, hi = 0, len(char_to_line) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if char_to_line[mid] <= cidx:
                lo = mid + 1
            else:
                hi = mid - 1
        return max(0, lo - 1)

    # Identify blocks by line ranges.
    def_starts = [char_idx_to_line_idx(m.start()) for m in matches]
    # Also consider any other top-level def/class to end blocks.
    top_decl = re.compile(r"^(def|class)\s+\w+\s*[\(:]", re.MULTILINE)
    decl_lines = sorted({i for i, ln in enumerate(lines) if top_decl.match(ln)})

    blocks: list[tuple[int, int]] = []
    for s in def_starts:
        # end at next top-level decl after s
        end = len(lines)
        for d in decl_lines:
            if d > s:
                end = d
                break
        blocks.append((s, end))

    def score(block: tuple[int, int]) -> int:
        s, e = block
        sc = 0
        for ln in lines[s:e]:
            t = ln.strip()
            if not t or t.startswith("#"):
                continue
            sc += 1
        return sc

    best = max(blocks, key=score)
    best_s, best_e = best

    # Remove other duplicate blocks while keeping everything else intact.
    remove_ranges = sorted([b for b in blocks if b != best], reverse=True)
    out_lines = lines[:]
    for s, e in remove_ranges:
        # blank out removed range
        for i in range(s, e):
            out_lines[i] = ""

    # Clean up consecutive blank lines
    cleaned: list[str] = []
    blank = False
    for ln in out_lines:
        if ln.strip() == "":
            if not blank:
                cleaned.append("")
            blank = True
        else:
            cleaned.append(ln)
            blank = False

    return "\n".join(cleaned).strip()


# --- (A) Import Completion ---

# Map: (usage_pattern, import_statement)
_IMPORT_RULES = [
    (r"\bheapq\b", "import heapq"),
    (r"\bcombinations\b", "from itertools import combinations"),
    (r"\bpermutations\b", "from itertools import permutations"),
    (r"\bproduct\b", "from itertools import product"),
    (r"\bCounter\b", "from collections import Counter"),
    (r"\bdefaultdict\b", "from collections import defaultdict"),
    (r"\bdeque\b", "from collections import deque"),
    (r"\bOrderedDict\b", "from collections import OrderedDict"),
    (r"\bmath\.", "import math"),
    (r"\bceil\b|\bfloor\b|\bsqrt\b|\bfabs\b|\bpow\b|\blog\b|\bsin\b|\bcos\b|\btan\b|\bpi\b", "import math"),
]

def _has_import(code: str, module: str, symbol: Optional[str] = None) -> bool:
    """Check if code already has the import."""
    if symbol:
        if re.search(rf"from\s+{re.escape(module)}\s+import\s+[^;\n]*\b{re.escape(symbol)}\b", code):
            return True
        if re.search(rf"import\s+{re.escape(module)}\b", code) and module == symbol:
            return True
    else:
        if re.search(rf"import\s+{re.escape(module)}\b", code):
            return True
        if re.search(rf"from\s+{re.escape(module)}\s+import", code):
            return True
    return False


def _add_missing_imports(code: str) -> str:
    """Add missing imports based on usage. Only add if not present."""
    imports_to_add = []

    if re.search(r"\bheapq\b", code) and not _has_import(code, "heapq"):
        imports_to_add.append("import heapq")

    it_symbols = [s for s in ["combinations", "permutations", "product"] if re.search(rf"\b{s}\b", code)]
    if it_symbols:
        missing = [s for s in it_symbols if not _has_import(code, "itertools", s)]
        if missing:
            imports_to_add.append(f"from itertools import {', '.join(missing)}")

    coll_symbols = [s for s in ["Counter", "defaultdict", "deque", "OrderedDict"] if re.search(rf"\b{s}\b", code)]
    if coll_symbols:
        missing = [s for s in coll_symbols if not _has_import(code, "collections", s)]
        if missing:
            imports_to_add.append(f"from collections import {', '.join(missing)}")

    needs_math = bool(re.search(r"\bmath\.", code))
    if needs_math and not _has_import(code, "math"):
        imports_to_add.append("import math")

    if not imports_to_add:
        return code

    # Insert after existing imports
    lines = code.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            insert_idx = i + 1
        elif insert_idx > 0 and s and not s.startswith("#"):
            break

    for imp in imports_to_add:
        if imp not in "\n".join(lines):
            lines.insert(insert_idx, imp)
            insert_idx += 1

    return "\n".join(lines)


# --- (B) Function Signature Enforcement ---


def _extract_first_function_name(code: str) -> Optional[str]:
    """Extract the name of the first defined function."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except SyntaxError:
        pass
    m = re.search(r"^\s*def\s+(\w+)\s*\(", code, re.MULTILINE)
    return m.group(1) if m else None


def _create_wrapper(inner_name: str, entry_point: str, code: str) -> str:
    """
    Create wrapper: def entry_point(...): return inner_name(...)
    We need to infer params from the inner function.
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == inner_name:
                args = [a.arg for a in node.args.args if a.arg != "self"]
                params = ", ".join(args)
                wrapper = f"\ndef {entry_point}({params}):\n    return {inner_name}({params})\n"
                return code + wrapper
    except SyntaxError:
        pass
    # Fallback: use *args, **kwargs
    wrapper = f"\ndef {entry_point}(*args, **kwargs):\n    return {inner_name}(*args, **kwargs)\n"
    return code + wrapper


def enforce_entry_point(code: str, entry_point: Optional[str]) -> str:
    """
    Ensure the generated code has a function matching entry_point.
    If mismatch: add wrapper def entry_point(...): return actual_func(...)
    """
    if not (entry_point or "").strip():
        return code

    first_func = _extract_first_function_name(code)
    if not first_func:
        return code
    if first_func == entry_point:
        return code

    return _create_wrapper(first_func, entry_point, code)


# --- Main API ---


def postprocess(
    code: str,
    entry_point: Optional[str] = None,
    *,
    strip_markdown: bool = True,
    add_imports: bool = True,
    enforce_fn: bool = True,
) -> str:
    """
    Apply full post-processing pipeline.

    Args:
        code: Raw generated code.
        entry_point: Expected function name (from problem).
        strip_markdown: Remove ``` blocks and extract code.
        add_imports: Add missing imports.
        enforce_fn: Enforce entry_point via wrapper if needed.

    Returns:
        Post-processed code.
    """
    if not (code or "").strip():
        return code
    # Skip postprocess for error outputs
    if code.strip().startswith("Error:"):
        return code.strip()

    result = code.strip()
    if strip_markdown:
        result = extract_code_block(result)

    # Remove common top-level test blocks that break evaluation.
    result = _strip_top_level_asserts_and_main(result)
    # Guard against accidental duplicate entry_point definitions.
    if entry_point:
        result = _dedupe_entry_point_defs(result, entry_point)

    if add_imports:
        result = _add_missing_imports(result)

    if enforce_fn and entry_point:
        result = enforce_entry_point(result, entry_point)

    return result.strip()
