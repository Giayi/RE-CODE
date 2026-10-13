#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强约束（soft 强化）：以软性提示方式强化代码生成约束。

不强制修改需求，而是在 prompt 中追加强约束提示，引导模型更好满足测试。
"""
from __future__ import annotations


# 强约束提示（soft 强化：作为 guidance 追加，非硬性改写）
SOFT_STRONG_CONSTRAINTS = """
# Strong Constraints (soft reinforcement - follow as guidance):
- MUST use the exact function name from the signature
- MUST match return type exactly (tuple vs list, order matters)
- Handle edge cases (empty input, boundary values)
- Respect ALL constraints stated in the problem
- Ensure deterministic output (avoid unordered set/list that may fail assertions)
- DO NOT introduce unnecessary helper functions unless required
- DO NOT include tests, asserts, or example invocations in the output
- DO NOT redefine the target function more than once
- Return only valid Python code. NO markdown. NO explanation.
"""


def get_soft_strong_constraints() -> str:
    """返回强约束（soft 强化）提示文本，用于追加到 user prompt。"""
    return SOFT_STRONG_CONSTRAINTS.strip()


def inject_soft_constraints(prompt: str) -> str:
    """
    在 prompt 末尾追加强约束（soft 强化）提示。

    Args:
        prompt: 当前 user prompt（含 docstring、def、test cases 等）

    Returns:
        追加强约束后的 prompt
    """
    if "Strong Constraints (soft reinforcement" in prompt:
        return prompt
    return prompt + "\n\n" + get_soft_strong_constraints()
