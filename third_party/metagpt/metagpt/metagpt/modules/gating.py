#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entropy-Aware Gating: decide whether to skip EGRCM for low-entropy (fully specified) tasks.

Low-entropy heuristics (skip EGRCM):
- contains assert statements
- contains function signature (def ...)
- contains explicit I/O examples (>>> or assert with expected output)

Generalizable: no dataset-specific logic.
"""
from __future__ import annotations

import re


def is_low_entropy(requirement: str) -> bool:
    """
    Detect if the requirement is "low-entropy" (fully specified).
    If True, EGRCM should be SKIPPED.
    If False, EGRCM should be USED.

    Heuristics:
    1. Contains assert statements (execution constraints)
    2. Contains function signature (def func_name)
    3. Contains explicit I/O examples (>>> or assert with ==)
    """
    if not (requirement or "").strip():
        return False

    text = requirement.strip()

    # Heuristic 1: assert statements
    if re.search(r"\bassert\s+", text):
        return True

    # Heuristic 2: function signature
    if re.search(r"\bdef\s+\w+\s*\(", text):
        return True

    # Heuristic 3: explicit I/O examples
    if ">>>" in text:
        return True
    if re.search(r"assert\s+.*==\s*", text):
        return True
    if re.search(r"(?:example|e\.g\.|for example)\s*:?\s*\w", text, re.I):
        return True

    return False


def should_skip_egrcm(requirement: str) -> bool:
    """
    Alias for is_low_entropy. Returns True if EGRCM should be skipped.
    """
    return is_low_entropy(requirement)
