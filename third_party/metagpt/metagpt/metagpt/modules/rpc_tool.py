#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RPC (Requirement-to-Prompt Compression) - compress structured requirements for code generation.

"""

from __future__ import annotations

import re
from typing import Any


def rpc_transform(code_requirement: str | dict[str, Any]) -> str:
    """
    Compress code-oriented requirement into a high-quality code generation prompt.

    Accepts output from Code Adapter (task, input, output, constraints) or
    raw PRQE output (product_goal, functional_requirements, etc.).

    Args:
        code_requirement: Code adapter output (dict with task/input/output/constraints)
            or structured requirement (dict) or plain text (str).

    Returns:
        Compact prompt string for code generation.

    Final prompt format:
        Write a Python function.

        Task:
        {task}

        Input:
        {input}

        Output:
        {output}

        Constraints:
        {constraints}

        Return only valid Python code.
        Do not include explanation.
        Do not include markdown.
    """
    task_desc = ""
    constraints: list[str] = []
    input_hint = ""
    output_hint = ""

    if isinstance(code_requirement, dict):
        # Adapter output format: task, input, output, constraints, function_signature, examples
        if "task" in code_requirement:
            task_desc = str(code_requirement.get("task", ""))
            input_hint = str(code_requirement.get("input", ""))
            output_hint = str(code_requirement.get("output", ""))
            raw = code_requirement.get("constraints", [])
            constraints = [str(c) for c in raw] if isinstance(raw, list) else []
            func_sig = str(code_requirement.get("function_signature", ""))
            func_name = str(code_requirement.get("function_name", ""))
            examples = str(code_requirement.get("examples", ""))
            return _build_prompt(
                task_desc, constraints, input_hint, output_hint,
                function_signature=func_sig, function_name=func_name, examples=examples,
            )
        else:
            task_desc, constraints, input_hint, output_hint = _extract_from_dict(
                code_requirement
            )
            return _build_prompt(task_desc, constraints, input_hint, output_hint)
    else:
        task_desc, constraints, input_hint, output_hint = _extract_from_text(
            str(code_requirement or "").strip()
        )
        if not task_desc:
            task_desc = str(code_requirement)[:500] if code_requirement else ""
        return _build_prompt(task_desc, constraints, input_hint, output_hint)


def _extract_from_dict(req: dict[str, Any]) -> tuple[str, list[str], str, str]:
    """Extract task, constraints, input/output from structured requirement dict."""
    task_parts: list[str] = []
    constraints: list[str] = []
    input_hint = ""
    output_hint = ""

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

    return task_desc, constraints, input_hint, output_hint


def _extract_from_text(text: str) -> tuple[str, list[str], str, str]:
    """Extract task, constraints, input/output from plain text (e.g. docstring)."""
    if not text:
        return "", [], "", ""

    task_desc = text
    constraints: list[str] = []
    input_hint = ""
    output_hint = ""

    # Try to parse docstring-style sections
    # Common patterns: "Args:", "Returns:", "Constraints:", "Input:", "Output:"
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
                input_hint = content.split("\n")[0][:200]
            elif "return" in header or "output" in header:
                output_hint = content.split("\n")[0][:200]
            elif "constraint" in header:
                for line in content.split("\n"):
                    line = line.strip().lstrip("-*•")
                    if line:
                        constraints.append(line[:200])
            i += 2

    # Fallback: look for constraint-like lines (ignore, case insensitive, etc.)
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

    return task_desc, constraints, input_hint, output_hint


def _build_prompt(
    task_desc: str,
    constraints: list[str],
    input_hint: str,
    output_hint: str,
    *,
    function_signature: str = "",
    function_name: str = "",
    examples: str = "",
) -> str:
    """Build the final compressed code prompt with contract constraints."""
    lines = [
        "Write a Python function.",
        "",
        "Task:",
        task_desc or "(See description above)",
        "",
    ]
    if function_name:
        lines.extend([f"Function name MUST be exactly: {function_name}", ""])
    if function_signature:
        lines.extend(["Function signature (use exactly):", function_signature, ""])
    if examples:
        lines.extend(["Examples (follow ALL examples strictly):", examples, ""])
    if input_hint:
        lines.extend(["Input:", input_hint, ""])
    if output_hint:
        lines.extend(["Output:", output_hint, ""])
    if constraints:
        lines.append("Constraints:")
        for c in constraints:
            lines.append(f"* {c}")
        lines.append("")
    lines.extend(
        [
            "MUST output full Python function (def + body).",
            "NO markdown. NO explanation. VALID syntax only.",
            "Ensure proper indentation (4 spaces for function body).",
        ]
    )
    return "\n".join(lines).strip()
