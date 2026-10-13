#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/3/16
@Desc    : Task Router - 根據需求內容判斷任務類型（PRD 或代碼生成）。

用於在 PRQE 之後路由到 PRD 流程或代碼流程。
RPC 僅在代碼流程中調用，PRD 評估實驗不經過此路由。
"""
from __future__ import annotations

import json
from typing import Any


# Keywords indicating programming/coding task
CODING_KEYWORDS = [
    "function",
    "algorithm",
    "implement",
    "python",
    "code",
    "API",
    "def ",
    "return",
    ">>>",
    "class ",
    "method",
    "parameter",
    "argument",
    "input",
    "output",
    "boolean",
    "string",
    "integer",
    "list",
    "dict",
    "array",
]


def detect_task_type(requirement: str | dict[str, Any]) -> str:
    """
    Determine whether the requirement is for product design or programming.

    Detection rules:
    - If contains keywords: function, implement, return, python, algorithm, >>>, etc.
      -> "code" (alias: "coding")
    - Otherwise -> "prd"

    Args:
        requirement: Structured requirement (dict) or plain text (str).

    Returns:
        "prd" - for product design / PRD generation
        "code" - for programming / code generation (use "coding" for backward compat)
    """
    text = _to_searchable_text(requirement)
    if not text:
        return "prd"

    text_lower = text.lower()
    # ">>>" is strong signal for code (docstring examples)
    if ">>>" in text:
        return "code"
    matches = sum(1 for kw in CODING_KEYWORDS if kw.lower() in text_lower)
    if matches >= 2:
        return "code"
    if matches >= 1 and _has_strong_coding_signal(text_lower):
        return "code"
    return "prd"


def _to_searchable_text(requirement: str | dict[str, Any]) -> str:
    """Convert requirement to searchable text."""
    if isinstance(requirement, str):
        return requirement
    if isinstance(requirement, dict):
        parts: list[str] = []
        for key in (
            "product_goal",
            "target_users",
            "functional_requirements",
            "non_functional_requirements",
            "constraints",
        ):
            val = requirement.get(key)
            if val is None:
                continue
            if isinstance(val, str):
                parts.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        parts.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(val, dict):
                parts.append(json.dumps(val, ensure_ascii=False))
        return " ".join(parts)
    return str(requirement)


def _has_strong_coding_signal(text: str) -> bool:
    """Check for strong indicators of coding task."""
    strong_signals = [
        "write a python",
        "implement a function",
        "create a function",
        "def ",
        "return ",
        "docstring",
        "input:",
        "output:",
        "parameters:",
        "returns:",
        ">>>",
    ]
    return any(s in text for s in strong_signals)
