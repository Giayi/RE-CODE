#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PRQE core logic: 7-dimension PRD quality evaluation.

from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field

from metagpt.configs.prqe_config import DEFAULT_WEIGHTS, PRQEConfig
from metagpt.logs import logger
from metagpt.utils.common import CodeParser


class PRQEEvaluationResult(BaseModel):
    """PRQE evaluation report."""

    overall_score: float = 0.0
    dimension_scores: dict[str, int] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    feedback_stage: str = ""


PRD_EVAL_PROMPT = """你是一位资深的产品需求文档（PRD）质量评估专家。请对以下 PRD 从 7 个维度进行评分（每维度 0-10 分）。

**评分维度说明：**
1. **completeness（完整性）**：是否包含目标用户、核心功能、非功能需求、系统约束、交互流程。
2. **clarity（清晰度）**：是否消除模糊/歧义。差例：「系统应有良好性能」；好例：「API 延迟 < 200ms」。
3. **feasibility（可行性）**：技术栈是否合适、目标是否现实、是否存在缺失的基础设施假设。
4. **consistency（一致性）**：是否存在需求冲突、矛盾功能、不一致的数据模型。
5. **testability（可测试性）**：是否可量化、可验证。好例：「支持 1000 并发用户」；差例：「系统应运行顺畅」。
6. **traceability（可追溯性）**：是否有需求 ID、功能-需求映射、需求-模块映射。
7. **atomicity（原子性）**：需求是否原子化。差例：「用户可注册并管理资料」；好例：「R1: 用户注册；R2: 资料更新」。

**PRD 内容：**
{prd_content}

**输出要求：** 仅返回一行 JSON，格式如下（不要 markdown 代码块、不要解释）：
{{"completeness": 8, "clarity": 7, "feasibility": 8, "consistency": 9, "testability": 7, "traceability": 6, "atomicity": 8, "issues": ["问题1", "问题2"], "improvement_suggestions": ["建议1", "建议2"]}}
"""

PRD_EVAL_PROMPT_SINGLE_REQ = """你是一位资深的需求质量评估专家。请对以下**单条需求**的结构化描述从 7 个维度进行评分（每维度 0-10 分）。

**说明**：这是单条原子级需求，非完整产品 PRD。评估时仅关注本条需求本身，不强制要求「目标用户」「系统约束」等产品级字段（可从需求陈述中推断即可）。

**评分维度说明：**
1. **completeness（完整性）**：是否包含需求陈述、依赖/约束、验收条件；表述是否完整可理解。
2. **clarity（清晰度）**：是否消除模糊/歧义。差例：「系统应有良好性能」；好例：「API 延迟 < 200ms」。
3. **feasibility（可行性）**：目标是否现实、是否存在缺失的基础设施假设。
4. **consistency（一致性）**：本条需求内部是否自洽，有无矛盾表述。
5. **testability（可测试性）**：是否可量化、可验证。好例：「支持 1000 并发用户」；差例：「系统应运行顺畅」。
6. **traceability（可追溯性）**：是否有需求 ID 或可追溯标识（可选）；与原始需求的对应关系是否清晰。
7. **atomicity（原子性）**：是否聚焦单一需求。差例：「用户可注册并管理资料」；好例：「用户可注册账户」。

**需求内容：**
{prd_content}

**输出要求：** 仅返回一行 JSON，格式如下（不要 markdown 代码块、不要解释）：
{{"completeness": 8, "clarity": 7, "feasibility": 8, "consistency": 9, "testability": 7, "traceability": 6, "atomicity": 8, "issues": ["问题1", "问题2"], "improvement_suggestions": ["建议1", "建议2"]}}
"""


def _parse_json_from_text(text: str) -> dict | None:
    """Parse JSON from text."""
    text = (text or "").strip()
    if "```" in text:
        if "```json" in text.lower():
            start = text.lower().find("```json")
            text = text[start + 7 :]
        else:
            start = text.find("```")
            text = text[start + 3 :]
        end = text.find("```")
        if end >= 0:
            text = text[:end]
    text = text.strip()

    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    return None


def _clamp_score(v: int | float) -> int:
    """Clamp score to 0-10."""
    try:
        n = int(v) if isinstance(v, (int, float)) else int(str(v).strip())
        return max(0, min(10, n))
    except (ValueError, TypeError):
        return 0


class PRDQualityEvaluator:
    """PRD quality evaluator."""

    def __init__(self, config: Optional[PRQEConfig] = None):
        self.config = config or PRQEConfig()
        self.weights = self.config.weights or dict(DEFAULT_WEIGHTS)

    def _compute_overall_score(self, scores: dict[str, int]) -> float:
        """Compute weighted overall score."""
        total = 0.0
        for dim, weight in self.weights.items():
            total += weight * scores.get(dim, 0)
        return round(total, 2)

    def _parse_llm_response(self, text: str) -> PRQEEvaluationResult:
        """Parse LLM JSON into PRQEEvaluationResult."""
        obj = _parse_json_from_text(text)
        if not obj:
            return PRQEEvaluationResult(
                overall_score=0.0,
                dimension_scores={},
                issues=["parse failed"],
                improvement_suggestions=[],
            )

        dims = [
            "completeness",
            "clarity",
            "feasibility",
            "consistency",
            "testability",
            "traceability",
            "atomicity",
        ]
        dimension_scores = {d: _clamp_score(obj.get(d, 0)) for d in dims}
        issues = obj.get("issues", [])
        if isinstance(issues, str):
            issues = [issues] if issues else []
        suggestions = obj.get("improvement_suggestions", [])
        if isinstance(suggestions, str):
            suggestions = [suggestions] if suggestions else []

        overall = self._compute_overall_score(dimension_scores)
        return PRQEEvaluationResult(
            overall_score=overall,
            dimension_scores=dimension_scores,
            issues=issues,
            improvement_suggestions=suggestions,
        )

    async def evaluate(
        self, prd_content: str, llm, single_requirement: bool | None = None
    ) -> PRQEEvaluationResult:
        """Evaluate PRD quality."""
        use_single = (
            single_requirement
            if single_requirement is not None
            else getattr(self.config, "single_requirement_mode", False)
        )
        prompt_tpl = PRD_EVAL_PROMPT_SINGLE_REQ if use_single else PRD_EVAL_PROMPT
        prompt = prompt_tpl.format(prd_content=prd_content[:15000])
        try:
            rsp = await llm.aask(prompt)
            result = self._parse_llm_response(rsp)
            logger.info(
                f"[PRQE] evaluation done: overall={result.overall_score}, "
                f"dims={result.dimension_scores}"
            )
            return result
        except Exception as e:
            logger.error(f"[PRQE] evaluation error: {e}")
            return PRQEEvaluationResult(
                overall_score=0.0,
                dimension_scores={},
                issues=[str(e)],
                improvement_suggestions=[],
            )

    def is_passed(self, result: PRQEEvaluationResult) -> bool:
        """Return True if score meets threshold."""
        return result.overall_score >= self.config.threshold
