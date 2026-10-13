#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Code Adapter - convert structured PRQE output to code-oriented task descriptions.

"""

from __future__ import annotations

import re
from typing import Any, Optional


def adapt_to_code_requirement(
    requirement: str | dict[str, Any],
    original_prompt: Optional[str] = None,
) -> dict[str, Any]:
    """
    Transform structured PRQE output into a concise code-oriented task description.

    Extracts the core programming task, preserves function signature and examples,
    removes product-level and user-level descriptions.

    Args:
        requirement: PRQE output - structured dict or plain text (e.g. docstring).
        original_prompt: Optional. Original prompt containing def and docstring.
            When provided, extracts function_signature and examples for preservation.

    Returns:
        Dict with keys: task, input, output, constraints, function_signature, examples
        {
            "task": "...",
            "input": "...",
            "output": "...",
            "constraints": ["...", ...],
            "function_signature": "def foo(...):",  # if original_prompt given
            "examples": ">>> foo(1)\n1",           # if found in docstring
        }
    """
    base = _adapt_from_dict(requirement) if isinstance(requirement, dict) else _adapt_from_text(str(requirement or "").strip())
    if original_prompt:
        sig = _extract_function_signature(original_prompt)
        req_text = requirement if isinstance(requirement, str) else str(base.get("task", ""))
        examples = _extract_examples(req_text, original_prompt)
        func_name = _extract_function_name(original_prompt)
        if sig:
            base["function_signature"] = sig
        if examples:
            base["examples"] = examples
        if func_name:
            base["function_name"] = func_name
    return base


def _extract_function_name(prompt: str) -> str:
    """Extract function name from def line (e.g. 'has_close_elements')."""
    if not prompt:
        return ""
    match = re.search(r"def\s+(\w+)\s*\(", prompt)
    return match.group(1) if match else ""


def _extract_function_signature(prompt: str) -> str:
    """Extract the def line from prompt (e.g. 'def has_close_elements(...) -> bool:' or 'def min_cost(cost, m, n)')."""
    if not prompt:
        return ""
    # HumanEval: def fn(...) -> Type:\n
    match = re.search(r"^(\s*def\s+\w+\s*\([^)]*\)\s*(?:->\s*[\w\[\],\s]+)?\s*:)", prompt, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # MBPP: def fn(args) without colon
    match = re.search(r"^(\s*def\s+\w+\s*\([^)]*\))", prompt, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_examples(requirement_text: str, original_prompt: str) -> str:
    """Extract >>> examples from docstring (input and output lines)."""
    text = original_prompt or requirement_text or ""
    examples: list[str] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines) and len(examples) < 5:
        line = lines[i]
        if ">>>" in line:
            ex = line.strip()
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith(">>>"):
                ex += "\n" + lines[i + 1].strip()
                i += 1
            if len(ex) < 400:
                examples.append(ex)
        i += 1
    return "\n".join(examples) if examples else ""


def _adapt_from_dict(req: dict[str, Any]) -> dict[str, Any]:
    """Extract code-oriented fields from structured PRQE output."""
    task_parts: list[str] = []
    constraints: list[str] = []
    input_hint = ""
    output_hint = ""

    # Prefer adapter output format if already adapted
    if "task" in req and isinstance(req.get("task"), str):
        return {
            "task": req["task"],
            "input": req.get("input", ""),
            "output": req.get("output", ""),
            "constraints": req.get("constraints", [])
            if isinstance(req.get("constraints"), list)
            else [],
        }

    # Extract from PRQE-style structured format
    if req.get("product_goal"):
        task_parts.append(str(req["product_goal"]))

    func_reqs = req.get("functional_requirements", [])
    if isinstance(func_reqs, list):
        for fr in func_reqs:
            if isinstance(fr, str):
                task_parts.append(fr)
            elif isinstance(fr, dict) and fr.get("description"):
                task_parts.append(str(fr["description"]))
    elif isinstance(func_reqs, str):
        task_parts.append(func_reqs)

    nfr = req.get("non_functional_requirements", [])
    if isinstance(nfr, list):
        for n in nfr:
            if isinstance(n, str):
                constraints.append(n)
            elif isinstance(n, dict) and n.get("description"):
                constraints.append(str(n["description"]))
    elif isinstance(nfr, str):
        constraints.append(nfr)

    raw_constraints = req.get("constraints", [])
    if isinstance(raw_constraints, list):
        for c in raw_constraints:
            if isinstance(c, str):
                constraints.append(c)
            elif isinstance(c, dict) and c.get("description"):
                constraints.append(str(c["description"]))
    elif isinstance(raw_constraints, str):
        constraints.append(raw_constraints)

    task_desc = "\n".join(task_parts).strip() if task_parts else ""

    return {
        "task": task_desc,
        "input": input_hint,
        "output": output_hint,
        "constraints": constraints,
    }


def _adapt_from_text(text: str) -> dict[str, Any]:
    """Extract code-oriented fields from plain text (e.g. docstring)."""
    if not text:
        return {"task": "", "input": "", "output": "", "constraints": []}

    task_desc = text
    constraints: list[str] = []
    input_hint = ""
    output_hint = ""

    # Try to parse docstring-style sections
    sections = re.split(
        r"\n\s*(Args?|Returns?|Constraints?|Inputs?|Outputs?|Parameters?)\s*:?\s*\n",
        text,
        flags=re.IGNORECASE,
    )
    if len(sections) > 1:
        task_desc = sections[0].strip()
        i = 1
        while i < len(sections) - 1:
            header = (sections[i] or "").strip().lower()
            content = (sections[i + 1] or "").strip()
            if "arg" in header or "param" in header or "input" in header:
                input_hint = content.split("\n")[0][:300]
            elif "return" in header or "output" in header:
                output_hint = content.split("\n")[0][:300]
            elif "constraint" in header:
                for line in content.split("\n"):
                    line = line.strip().lstrip("-*•")
                    if line:
                        constraints.append(line[:200])
            i += 2

    # Fallback: look for constraint-like lines
    if not constraints:
        constraint_keywords = [
            r"ignore\s+(\w+)",
            r"case\s+insensitive",
            r"case\s+sensitive",
            r"(\w+)\s+must\s+be",
            r"do\s+not\s+([^.\n]+)",
            r"assume\s+([^.\n]+)",
        ]
        for pat in constraint_keywords:
            for m in re.finditer(pat, text, re.IGNORECASE):
                constraints.append(m.group(0).strip()[:150])

    return {
        "task": task_desc[:2000] if task_desc else "",
        "input": input_hint,
        "output": output_hint,
        "constraints": constraints,
    }
