#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PRQE (Product Requirement Quality Evaluator) configuration."""
from pydantic import BaseModel, Field


DEFAULT_WEIGHTS = {
    "completeness": 0.2,
    "clarity": 0.2,
    "feasibility": 0.2,
    "consistency": 0.15,
    "testability": 0.1,
    "traceability": 0.1,
    "atomicity": 0.05,
}


class PRQEConfig(BaseModel):
    """PRQE module configuration."""

    enabled: bool = False
    requirement_level: bool = True
    threshold: float = 8.0
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    max_refinement_rounds: int = 3
    single_requirement_mode: bool = False
