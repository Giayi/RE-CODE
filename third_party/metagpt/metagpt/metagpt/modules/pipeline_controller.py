#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline Controller - route tasks to PRD or code generation after PRQE.

Flow:
  Requirement -> EGRCM -> PRQE -> Task Router
  |-- PRD Flow    -> PRD Generation
  `-- Code Flow   -> Code Adapter -> RPC -> Code Generation

Experiment 2 bypasses this controller (Requirement -> EGRCM -> PRQE -> PRD).
Adapter and RPC are used only for code-generation tasks.
"""

from __future__ import annotations

from typing import Any, Literal

from metagpt.modules.code_adapter import adapt_to_code_requirement
from metagpt.modules.rpc_tool import rpc_transform
from metagpt.modules.task_router import detect_task_type


TaskType = Literal["prd", "coding"]


def route_after_prqe(
    enhanced_requirement: str | dict[str, Any],
    *,
    downstream_task: TaskType | None = None,
    force_prd: bool = False,
) -> tuple[TaskType, str | None]:
    """
    After PRQE, determine route and optionally produce RPC-compressed prompt.

    Args:
        enhanced_requirement: Output from PRQE (refined requirement).
        downstream_task: If provided, use this instead of detection.
            "prd" -> PRD flow, no RPC.
            "coding" -> Code flow, apply RPC.
        force_prd: If True, always return "prd" (e.g. for Experiment 2).

    Returns:
        (task_type, rpc_prompt_or_none)
        - For "prd": ( "prd", None )
        - For "coding": ( "coding", rpc_compressed_prompt )
    """
    if force_prd:
        return ("prd", None)

    raw_type = downstream_task or detect_task_type(enhanced_requirement)
    task_type = raw_type if raw_type == "prd" else "coding"  # normalize "code" -> "coding"

    if task_type == "prd":
        return ("prd", None)

    # code/coding: Adapter → RPC (route_after_prqe has no original_prompt; use enhanced only)
    code_req = adapt_to_code_requirement(enhanced_requirement)
    prompt = rpc_transform(code_req)
    return ("coding", prompt)


def get_code_prompt_via_rpc(
    enhanced_requirement: str | dict[str, Any],
    original_prompt: str | None = None,
) -> str:
    """
    Get RPC-compressed prompt for code generation.

    Use when downstream task is known to be code generation (e.g. Experiment 3).
    Skips task detection. Applies Code Adapter + RPC pipeline.

    Args:
        enhanced_requirement: Output from PRQE.
        original_prompt: Optional. Original problem prompt (with def and docstring).
            When provided, Adapter preserves function signature and examples.

    Returns:
        Compact code-generation prompt (Adapter → RPC).
    """
    code_req = adapt_to_code_requirement(enhanced_requirement, original_prompt)
    return rpc_transform(code_req)
