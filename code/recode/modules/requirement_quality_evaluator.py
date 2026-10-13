#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PRQE requirement-level quality control and refinement (before PRD generation).

from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from metagpt.configs.prqe_config import DEFAULT_WEIGHTS, PRQEConfig
from metagpt.logs import logger


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if exception is a rate-limit (429) error."""
    msg = str(exc).lower()
    return "429" in msg or "限流" in msg or "rate limit" in msg


REQUIREMENT_STRUCTURE_TEMPLATE = """需求應遵循以下結構：
- Requirement ID: 唯一標識（如 REQ-001）
- Actor: 執行者
- Action: 動作
- Object: 對象
- Constraints: 約束條件（可量化）
- Acceptance Criteria: 驗收標準（可驗證）

示例：
REQ-001
Actor: User
Action: upload
Object: files
Constraint: file size must be less than 10MB
Acceptance Criteria: upload succeeds within 2 seconds
"""


class RequirementIssue(BaseModel):
    """Single requirement issue."""

    type: str = Field(description="Issue type")
    description: str = Field(description="Issue description")


class RequirementEvaluationResult(BaseModel):
    """Requirement quality evaluation result."""

    scores: dict[str, int] = Field(default_factory=dict)
    overall_score: float = 0.0
    issues: list[RequirementIssue] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)


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


REQUIREMENT_EVAL_PROMPT = """你是一位資深的需求質量評估專家。請對以下**需求**（單條或需求列表）從 7 個維度進行評分（每維度 0-10 分）。

**需求應遵循的結構規範：**
{structure_template}

**評分維度說明：**
1. **completeness（完整性）**：是否包含 Requirement ID、Actor、Action、Object、Constraints、Acceptance Criteria。
2. **clarity（清晰度）**：是否消除模糊/歧義。差例：「系統應有良好性能」；好例：「API 延遲 < 200ms」。
3. **feasibility（可行性）**：目標是否現實、是否存在缺失的基礎設施假設。
4. **consistency（一致性）**：需求內部是否自洽，與其他需求有無衝突。
5. **testability（可測試性）**：是否可量化、可驗證。
6. **traceability（可追溯性）**：是否有需求 ID、與原始需求的對應關係是否清晰。
7. **atomicity（原子性）**：是否聚焦單一需求。差例：「用戶可註冊並管理資料」；好例：「用戶可註冊賬戶」。

**需求內容：**
{requirement_content}

**輸出要求：** 僅返回一行 JSON，格式如下（不要 markdown 代碼塊、不要解釋）：
{{"scores": {{"completeness": 8, "clarity": 7, "feasibility": 8, "consistency": 9, "testability": 7, "traceability": 5, "atomicity": 8}}, "overall_score": 7.4}}
"""


REQUIREMENT_ISSUE_DETECT_PROMPT = """你是一位資深的需求分析專家。請檢測以下需求中的問題。

**需檢測的問題類型：**
1. missing_requirement_elements - 缺少需求要素（ID、Actor、Action、Object、Constraints、Acceptance Criteria）
2. ambiguous_descriptions - 模糊描述（如「快速」「良好」等不可量化詞彙）
3. missing_measurable_constraints - 缺少可量化約束
4. missing_acceptance_criteria - 缺少驗收標準
5. missing_requirement_id - 缺少需求 ID
6. conflicting_requirements - 需求之間或內部衝突

**需求內容：**
{requirement_content}

**輸出要求：** 僅返回一行 JSON，格式如下（不要 markdown 代碼塊、不要解釋）：
{{"issues": [{{"type": "traceability", "description": "Requirement does not contain a unique ID"}}, {{"type": "ambiguity", "description": "The phrase 'fast response' is not measurable"}}]}}
若無問題，返回 {{"issues": []}}
"""


REQUIREMENT_FEEDBACK_PROMPT = """你是一位資深的需求工程師。根據以下需求及其評估結果，生成**可直接用於需求改寫**的改進建議。

**需求內容：**
{requirement_content}

**評估結果：**
- 各維度分數：{scores}
- 發現的問題：{issues}

**輸出要求：** 僅返回一行 JSON，格式如下（不要 markdown 代碼塊、不要解釋）：
{{"improvement_suggestions": ["Add a unique requirement ID (e.g., REQ-001)", "Replace vague terms like 'fast response' with measurable constraints", "Add acceptance criteria for validation"]}}
"""


REQUIREMENT_REFINE_PROMPT = """你是一位资深的需求工程师。请根据以下改进建议，对需求进行改写，输出优化后的需求。

**原始需求：**
{requirement_content}

**改进建议：**
{feedback}

**输出要求：**
1. 严格遵循结构：Requirement ID、Actor、Action、Object、Constraints、Acceptance Criteria
2. 仅输出优化后的需求文本，不要额外解释或 markdown 标记
3. 使用简体中文输出
"""



class RequirementQualityEvaluator:
    """7-dimension requirement quality scorer."""

    def __init__(self, config: Optional[PRQEConfig] = None):
        self.config = config or PRQEConfig()
        self.weights = self.config.weights or dict(DEFAULT_WEIGHTS)

    def _compute_overall_score(self, scores: dict[str, int]) -> float:
        """Compute weighted overall score."""
        total = 0.0
        for dim, weight in self.weights.items():
            total += weight * scores.get(dim, 0)
        return round(total, 2)

    def _parse_llm_response(self, text: str) -> RequirementEvaluationResult:
        """Parse LLM JSON response."""
        obj = _parse_json_from_text(text)
        if not obj:
            return RequirementEvaluationResult(
                scores={},
                overall_score=0.0,
                issues=[],
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
        scores_obj = obj.get("scores", obj)
        dimension_scores = {
            d: _clamp_score(scores_obj.get(d, 0)) for d in dims
        }
        overall = obj.get("overall_score")
        if overall is None:
            overall = self._compute_overall_score(dimension_scores)
        else:
            overall = round(float(overall), 2)

        issues_raw = obj.get("issues", [])
        issues = []
        for item in issues_raw if isinstance(issues_raw, list) else []:
            if isinstance(item, dict):
                issues.append(
                    RequirementIssue(
                        type=str(item.get("type", "unknown")),
                        description=str(item.get("description", "")),
                    )
                )
            elif isinstance(item, str):
                issues.append(RequirementIssue(type="general", description=item))

        return RequirementEvaluationResult(
            scores=dimension_scores,
            overall_score=overall,
            issues=issues,
            improvement_suggestions=obj.get("improvement_suggestions", []),
        )

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    async def evaluate(
        self, requirement: str, llm
    ) -> RequirementEvaluationResult:
        """Evaluate requirement quality."""
        prompt = REQUIREMENT_EVAL_PROMPT.format(
            structure_template=REQUIREMENT_STRUCTURE_TEMPLATE,
            requirement_content=requirement[:15000],
        )
        try:
            rsp = await llm.aask(prompt)
            result = self._parse_llm_response(rsp)
            logger.info(
                f"[PRQE] requirement evaluation done: overall={result.overall_score}, "
                f"scores={result.scores}"
            )
            return result
        except Exception as e:
            if _is_rate_limit_error(e):
                raise
            logger.error(f"[PRQE] requirement evaluation error: {e}")
            return RequirementEvaluationResult(
                scores={},
                overall_score=0.0,
                issues=[RequirementIssue(type="error", description=str(e))],
                improvement_suggestions=[],
            )

    def is_passed(self, result: RequirementEvaluationResult) -> bool:
        """Return True if score meets threshold."""
        return result.overall_score >= self.config.threshold


class RequirementIssueDetector:
    """Detect requirement quality issues."""

    def _parse_issues(self, text: str) -> list[RequirementIssue]:
        """Parse issue list from LLM response."""
        obj = _parse_json_from_text(text)
        if not obj:
            return []

        issues_raw = obj.get("issues", [])
        issues = []
        for item in issues_raw if isinstance(issues_raw, list) else []:
            if isinstance(item, dict):
                issues.append(
                    RequirementIssue(
                        type=str(item.get("type", "unknown")),
                        description=str(item.get("description", "")),
                    )
                )
            elif isinstance(item, str):
                issues.append(RequirementIssue(type="general", description=item))
        return issues

    async def detect(self, requirement: str, llm) -> list[RequirementIssue]:
        """Detect issues in a requirement."""
        prompt = REQUIREMENT_ISSUE_DETECT_PROMPT.format(
            requirement_content=requirement[:15000],
        )
        try:
            rsp = await llm.aask(prompt)
            issues = self._parse_issues(rsp)
            logger.info(f"[PRQE] issue detection done: found {len(issues)}  issues")
            return issues
        except Exception as e:
            logger.error(f"[PRQE] issue detection error: {e}")
            return [RequirementIssue(type="error", description=str(e))]


class RequirementFeedbackGenerator:
    """Generate structured refinement feedback."""

    def _parse_suggestions(self, text: str) -> list[str]:
        """Parse improvement suggestions from LLM response."""
        obj = _parse_json_from_text(text)
        if not obj:
            return []

        suggestions = obj.get("improvement_suggestions", [])
        if isinstance(suggestions, str):
            return [suggestions] if suggestions else []
        return [str(s) for s in suggestions if s]

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    async def generate(
        self,
        requirement: str,
        evaluation_result: RequirementEvaluationResult,
        llm,
    ) -> list[str]:
        """Generate improvement suggestions."""
        issues_str = [
            f"- {i.type}: {i.description}" for i in evaluation_result.issues
        ]
        prompt = REQUIREMENT_FEEDBACK_PROMPT.format(
            requirement_content=requirement[:8000],
            scores=evaluation_result.scores,
            issues="\n".join(issues_str) if issues_str else "none",
        )
        try:
            rsp = await llm.aask(prompt)
            suggestions = self._parse_suggestions(rsp)
            if evaluation_result.improvement_suggestions and not suggestions:
                suggestions = evaluation_result.improvement_suggestions
            logger.info(f"[PRQE] feedback generation done: {len(suggestions)}  suggestions")
            return suggestions
        except Exception as e:
            if _is_rate_limit_error(e):
                raise
            logger.error(f"[PRQE] feedback generation error: {e}")
            return evaluation_result.improvement_suggestions or []


class RequirementRefiner:
    """Refine requirements from PRQE feedback."""

    @retry(
        retry=retry_if_exception(_is_rate_limit_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    async def refine(
        self,
        requirement: str,
        feedback: list[str] | RequirementEvaluationResult,
        llm,
    ) -> str:
        """Refine requirement from feedback."""
        if isinstance(feedback, RequirementEvaluationResult):
            feedback_list = feedback.improvement_suggestions
        else:
            feedback_list = feedback or []

        feedback_text = "\n".join(f"- {s}" for s in feedback_list)
        if not feedback_text:
            return requirement

        prompt = REQUIREMENT_REFINE_PROMPT.format(
            requirement_content=requirement[:15000],
            feedback=feedback_text,
        )
        try:
            rsp = await llm.aask(prompt)
            refined = (rsp or "").strip()
            if refined:
                logger.info("[PRQE] requirement refinement done")
                return refined
        except Exception as e:
            if _is_rate_limit_error(e):
                raise
            logger.error(f"[PRQE] requirement refinement error: {e}")
        return requirement


# ============ PRQE Code-Mode Hints (Experiment 3 only, no rewrite) ============


def detect_mode(prompt: str) -> str:
    """Detect HumanEval vs MBPP mode from prompt structure.

    MBPP: short requirement (< 2 lines) or no examples.
    HumanEval: longer docstring with examples.
    """
    if not (prompt or "").strip():
        return "HumanEval"
    prompt = prompt.strip()
    # Extract requirement part (before first "def ")
    req_part = prompt
    if "def " in prompt:
        idx = prompt.find("def ")
        req_part = prompt[:idx].strip()
    req_lines = [l.strip() for l in req_part.split("\n") if l.strip()]
    num_lines = len(req_lines)
    has_examples = (
        ">>>" in prompt
        or "example" in prompt.lower()
        or "for example" in prompt.lower()
        or "e.g." in prompt.lower()
    )
    if num_lines < 2 or not has_examples:
        return "MBPP"
    return "HumanEval"


# Static hints for HumanEval mode (semantic enhancement)
_HUMANEVAL_CODE_HINTS = """- Handle edge cases (empty input, boundary values)
- Respect all constraints stated in the problem
- Ensure correctness for all valid inputs"""

# Static hints for MBPP mode (test-alignment enhancement)
_MBPP_CODE_HINTS = """- MUST use the exact function name from the signature
- MUST match return type exactly (tuple vs list, order matters)
- DO NOT introduce unnecessary helper functions unless required by the problem
- Ensure deterministic output (avoid unordered set/list that may fail assertions)"""

# Test alignment hints appended in MBPP mode
_MBPP_TEST_ALIGNMENT_HINTS = """- Follow function signature exactly
- Return type must match expected tests
- Avoid randomness or unordered outputs"""


async def generate_code_hints(
    prompt: str,
    mode: str,
    llm=None,
    config: Optional[PRQEConfig] = None,
    requirement_override: Optional[str] = None,
) -> str:
    """Generate PRQE code-mode hints for Experiment 3. Does NOT rewrite the requirement.

    Mode "HumanEval": semantic enhancement (edge cases, constraints, correctness).
    Mode "MBPP": test-alignment enhancement (exact function name, return type, no extra helpers).
    When llm is provided and mode is HumanEval, merges LLM-based quality suggestions.
    requirement_override: when set (e.g. EGRCM-refined docstring), use for LLM analysis.
    Adapter/RPC must NOT receive PRQE output; hints are side-channel only.
    """
    if not (prompt or "").strip():
        return ""
    if mode == "MBPP":
        return _MBPP_CODE_HINTS
    # HumanEval mode
    base_hints = _HUMANEVAL_CODE_HINTS
    if llm:
        try:
            req = requirement_override
            if not req:
                req = prompt
                if "def " in prompt:
                    idx = prompt.find("def ")
                    req = prompt[:idx].strip()
                for pattern in [r'"""(.*?)"""', r"'''(.*?)'''"]:
                    m = re.search(pattern, req, re.S)
                    if m:
                        req = m.group(1).strip()
                        break
            if req:
                extra = await generate_prqe_hints(req, llm, config)
                if extra:
                    base_hints = base_hints + "\n\n" + extra
        except Exception as e:
            logger.warning(f"[PRQE] generate_code_hints HumanEval LLM fallback: {e}")
    return base_hints


def get_test_alignment_hints() -> str:
    """Test alignment hints for MBPP mode. Append after PRQE hints."""
    return _MBPP_TEST_ALIGNMENT_HINTS


async def generate_prqe_hints(
    requirement: str,
    llm,
    config: Optional[PRQEConfig] = None,
) -> str:
    """Generate PRQE hints for optional guidance. Does NOT rewrite the requirement.

    Used by Experiment 3 code generation pipeline. Output is injected as side-channel
    guidance; original docstring is preserved. Experiment 1 and 2 use refine_requirement.
    """
    if not (requirement or "").strip():
        return ""
    cfg = config or PRQEConfig()
    evaluator = RequirementQualityEvaluator(cfg)
    result = await evaluator.evaluate(requirement, llm)

    feedback_gen = RequirementFeedbackGenerator()
    suggestions = await feedback_gen.generate(requirement, result, llm)
    if not suggestions:
        suggestions = result.improvement_suggestions or []

    lines = [
        f"Quality score: {result.overall_score}/10",
        f"Dimensions: {result.scores}",
    ]
    if result.issues:
        lines.append("Issues detected:")
        for i in result.issues:
            lines.append(f"  - {i.type}: {i.description}")
    if suggestions:
        lines.append("Suggestions (optional guidance):")
        for s in suggestions:
            lines.append(f"  - {s}")
    return "\n".join(lines)



async def refine_requirement(
    requirement: str,
    feedback: list[str] | RequirementEvaluationResult,
    llm,
) -> str:
    """Refine requirement; convenience wrapper around RequirementRefiner.refine."""
    refiner = RequirementRefiner()
    return await refiner.refine(requirement, feedback, llm)
