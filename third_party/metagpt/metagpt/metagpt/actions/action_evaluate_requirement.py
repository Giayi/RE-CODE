#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/3/11
@Desc    : PRQE 需求級評估 Action：在 PRD 生成之前對需求進行質量評估與精煉。

新流程：refined_requirement → PRQE.evaluate → feedback → refinement → refined_requirement_v2 → PrepareDocuments → WritePRD
"""
from __future__ import annotations

from typing import List, Optional

from metagpt.actions import Action, UserRequirement
from metagpt.actions.action_reduce_entropy import ActionReduceEntropy
from metagpt.configs.prqe_config import PRQEConfig
from metagpt.logs import logger
from metagpt.modules.requirement_quality_evaluator import (
    RequirementEvaluationResult,
    RequirementFeedbackGenerator,
    RequirementIssueDetector,
    RequirementQualityEvaluator,
    RequirementRefiner,
)
from metagpt.schema import AIMessage, Message
from metagpt.utils.common import any_to_str


class ActionEvaluateRequirement(Action):
    """需求級 PRQE：評估需求質量，未通過時觸發精煉。

    接收 ActionReduceEntropy 或 UserRequirement 輸出，對需求進行評估。
    通過則輸出供 PrepareDocuments 使用；未通過則觸發 ActionRefineRequirement。
    """

    name: str = "ActionEvaluateRequirement"
    prqe_config: Optional[PRQEConfig] = None

    def __init__(self, prqe_config: Optional[PRQEConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.prqe_config = prqe_config or PRQEConfig()

    def _get_requirement_content(self, with_messages: List[Message]) -> str:
        """從消息中提取需求內容。優先使用 ActionRefineRequirement 的精煉結果，其次 ActionReduceEntropy，最後 UserRequirement。"""
        for msg in reversed(with_messages or []):
            cause = getattr(msg, "cause_by", None)
            if cause == any_to_str(ActionRefineRequirement):
                return (msg.content or "").strip()
            if cause == any_to_str(ActionReduceEntropy):
                return (msg.content or "").strip()
            if cause == any_to_str(UserRequirement):
                return (msg.content or "").strip()
        if with_messages:
            return (with_messages[-1].content or "").strip()
        return ""

    def _count_refinement_rounds(self, with_messages: List[Message]) -> int:
        """統計已執行的精煉輪數。"""
        return sum(
            1
            for m in with_messages or []
            if getattr(m, "cause_by", None) == any_to_str(ActionRefineRequirement)
        )

    async def run(
        self,
        with_messages: Optional[List[Message]] = None,
        **kwargs,
    ) -> Optional[AIMessage]:
        """評估需求質量，通過則輸出供 PrepareDocuments，未通過則觸發精煉。"""
        if not with_messages:
            logger.warning("[PRQE-Req] 無輸入消息")
            return None

        requirement = self._get_requirement_content(with_messages)
        if not requirement:
            logger.warning("[PRQE-Req] 未找到需求內容")
            return None

        refinement_count = self._count_refinement_rounds(with_messages)
        if refinement_count >= self.prqe_config.max_refinement_rounds:
            logger.warning(
                f"[PRQE-Req] 已達最大精煉輪數 ({self.prqe_config.max_refinement_rounds})，透傳當前需求"
            )
            return self._output_for_prepare_documents(requirement)

        evaluator = RequirementQualityEvaluator(self.prqe_config)
        result = await evaluator.evaluate(requirement, self.llm)

        if evaluator.is_passed(result):
            logger.info(
                f"[PRQE-Req] 通過 (score={result.overall_score})，輸出至 PrepareDocuments"
            )
            return self._output_for_prepare_documents(requirement)

        logger.info(
            f"[PRQE-Req] 未通過 (score={result.overall_score})，觸發需求精煉"
        )
        return AIMessage(
            content=f"需求質量未達閾值 ({result.overall_score} < {self.prqe_config.threshold})",
            instruct_content=AIMessage.create_instruct_value(
                kvs={
                    "evaluation_result": result.model_dump(),
                    "requirement": requirement,
                    "refinement_round": refinement_count + 1,
                },
                class_name="PRQERequirementEvaluationOutput",
            ),
            cause_by=any_to_str(ActionEvaluateRequirement),
            send_to="Alice",
        )

    def _output_for_prepare_documents(self, requirement: str) -> AIMessage:
        """輸出供 PrepareDocuments 使用的消息。"""
        return AIMessage(
            content=requirement,
            instruct_content=AIMessage.create_instruct_value(
                kvs={
                    "requirement_content": requirement,
                    "prqe_passed": True,
                },
                class_name="PRQERequirementPassedOutput",
            ),
            cause_by=any_to_str(ActionEvaluateRequirement),
            send_to="Alice",
        )


class ActionRefineRequirement(Action):
    """根據 PRQE 反饋精煉需求。"""

    name: str = "ActionRefineRequirement"
    prqe_config: Optional[PRQEConfig] = None

    def __init__(self, prqe_config: Optional[PRQEConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.prqe_config = prqe_config or PRQEConfig()

    async def run(
        self,
        with_messages: Optional[List[Message]] = None,
        **kwargs,
    ) -> Optional[AIMessage]:
        """根據評估結果精煉需求，輸出供 ActionEvaluateRequirement 再次評估。"""
        if not with_messages:
            logger.warning("[PRQE-Refine] 無輸入消息")
            return None

        eval_msg = None
        for msg in reversed(with_messages or []):
            if getattr(msg, "cause_by", None) == any_to_str(ActionEvaluateRequirement):
                eval_msg = msg
                break

        if not eval_msg or not eval_msg.instruct_content:
            logger.warning("[PRQE-Refine] 未找到 ActionEvaluateRequirement 輸出")
            return None

        ic = eval_msg.instruct_content
        kvs = ic.model_dump() if hasattr(ic, "model_dump") else dict(ic)
        requirement = kvs.get("requirement", "")
        eval_result_dict = kvs.get("evaluation_result", {})

        if not requirement:
            logger.warning("[PRQE-Refine] 需求內容為空")
            return None

        from metagpt.modules.requirement_quality_evaluator import (
            RequirementEvaluationResult,
            RequirementRefiner,
        )

        eval_result = RequirementEvaluationResult(**eval_result_dict)
        refiner = RequirementRefiner()
        feedback_gen = RequirementFeedbackGenerator()

        suggestions = await feedback_gen.generate(
            requirement, eval_result, self.llm
        )
        if not suggestions:
            suggestions = eval_result.improvement_suggestions

        refined = await refiner.refine(requirement, suggestions, self.llm)

        logger.info("[PRQE-Refine] 需求精煉完成，將再次評估")

        return AIMessage(
            content=refined,
            instruct_content=AIMessage.create_instruct_value(
                kvs={"requirement_content": refined},
                class_name="PRQERefinedRequirementOutput",
            ),
            cause_by=any_to_str(ActionRefineRequirement),
            send_to="Alice",
        )
