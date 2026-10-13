#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CP-EGRCM (Contract-Preserving EGRCM) - contract-preserving refinement for code tasks.

Used for code generation only. Preserves signature, return type, and examples.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

CP_EGRCM_PROMPT = """You are refining a programming task specification.

STRICT RULES:
1. DO NOT change the function signature.
2. DO NOT change return type.
3. DO NOT add new constraints.
4. DO NOT remove or modify examples (>>>).
5. DO NOT introduce validation or error handling unless explicitly required.
6. DO NOT translate the original text.
7. DO NOT rename the function.

Your goal: Clarify the requirement WITHOUT changing its semantics.

Output format:

Task:
(restated clearly, but semantically identical)

Input:
(original definition)

Output:
(original definition)

Constraints:
(ONLY constraints explicitly mentioned)

Examples:
(copy ALL original examples exactly)

---

Original requirement:

{requirement}

---

Output (follow the format above, no extra text):"""


async def cp_egrcm_process(requirement: str, llm) -> str:
    """
    Contract-Preserving Requirement Refinement for code tasks.

    Refines the requirement while preserving:
    - Function signature
    - Return type
    - All examples (>>>)
    - Original language
    - No new constraints or validation

    Args:
        requirement: Original requirement text (e.g. docstring).
        llm: LLM instance for refinement.

    Returns:
        Refined requirement string, or original if empty/failed.
    """
    if not (requirement or "").strip():
        return requirement or ""

    prompt = CP_EGRCM_PROMPT.format(requirement=(requirement or "").strip())
    try:
        result = await llm.aask(prompt)
        refined = (result or "").strip()
        if refined:
            return refined
    except Exception:
        pass
    return requirement or ""
