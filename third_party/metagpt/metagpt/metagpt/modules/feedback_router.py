#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Map PRQE issue types to entropy dialogue stages.

"""

from __future__ import annotations


ISSUE_TO_STAGE_MAP = {
    "feature_decomposition": [
        "缺失", "缺少", "遗漏", "missing", "incomplete",
        "功能范围", "scope", "feature",
        "must_features", "optional_features",
    ],
    "requirement_quantification": [
        "模糊", "歧义", "不明确", "ambiguous", "vague",
        "量化", "具体", "quantification",
    ],
    "product_boundary": [
        "冲突", "矛盾", "不一致", "conflict", "contradiction",
        "boundary", "exclusion", "排除",
    ],
    "priority_confirmation": [
        "优先级", "priority", "success_metrics",
        "度量", "metric",
    ],
}


def route(issues: list[str], suggestions: list[str]) -> str:
    """Route feedback to an entropy dialogue stage."""
    combined = " ".join(issues + suggestions).lower()
    scores: dict[str, int] = {stage: 0 for stage in ISSUE_TO_STAGE_MAP}

    for stage, keywords in ISSUE_TO_STAGE_MAP.items():
        for kw in keywords:
            if kw.lower() in combined:
                scores[stage] += 1

    if max(scores.values()) == 0:
        return "feature_decomposition"

    return max(scores, key=scores.get)
