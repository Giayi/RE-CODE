#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test-Aware Prompt Injection: append test cases to prompt after PRQE output.

Aligns generation with execution constraints by explicitly including tests.
"""
from __future__ import annotations


TEST_INJECTION_PREFIX = (
    "\n\nYou must implement a function that satisfies the following tests:\n\n"
)


def inject_test_cases(prompt: str, test_cases: str | None) -> str:
    """
    If test cases exist, append them to the prompt.

    Format: "You must implement a function that satisfies the following tests:\n\n<tests>"

    Args:
        prompt: Current prompt (after PRQE enhancements).
        test_cases: Raw test code (e.g. problem["test"]), or None.

    Returns:
        Prompt with test cases appended if they exist.
    """
    if not (test_cases or "").strip():
        return prompt

    tests = test_cases.strip()
    # Avoid duplicate injection
    if "satisfies the following tests" in prompt:
        return prompt

    return prompt + TEST_INJECTION_PREFIX + tests


def get_test_cases_from_problem(problem: dict) -> str | None:
    """
    Extract test cases from problem dict (HumanEval/MBPP format).
    """
    return problem.get("test")
