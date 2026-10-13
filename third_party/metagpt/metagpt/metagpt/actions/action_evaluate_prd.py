#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/3/9
@Desc    : PRQE Action：评估 PRD 质量，决定是否通过或触发精炼循环。

当 PRQE 启用时，本 Action 内部调用 WritePRD，避免未通过评估的 PRD 被发布给 Architect。
"""
from __future__ import annotations

from typing import List, Optional

from metagpt.actions import Action
from metagpt.actions.write_prd import WritePRD
from metagpt.configs.prqe_config import PRQEConfig
from metagpt.logs import logger
from metagpt.modules.prd_quality_evaluator import PRDQualityEvaluator, PRQEEvaluationResult
from metagpt.schema import AIMessage, Document, Message
from metagpt.utils.common import any_to_str


class ActionEvaluatePRD(Action):
    """评估 PRD 质量，根据分数决定透传或触发精炼。

    接收 PrepareDocuments 或 ActionGenerateFeedback 输出时，内部调用 WritePRD 生成 PRD，
    评估后：score >= threshold 则透传供 Architect 接收；否则触发 ActionGenerateFeedback。
    """

    name: str = "ActionEvaluatePRD"
    prqe_config: Optional[PRQEConfig] = None

    def __init__(self, prqe_config: Optional[PRQEConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.prqe_config = prqe_config or PRQEConfig()

    async def run(
        self,
        with_messages: Optional[List[Message]] = None,
        **kwargs,
    ) -> Optional[AIMessage]:
        """评估 PRD，返回透传消息或精炼触发消息。"""
        if not with_messages:
            logger.warning("[PRQE] 无输入消息")
            return None

        # 检查精炼轮数，防止死循环
        refinement_count = sum(
            1
            for m in with_messages or []
            if getattr(m, "cause_by", None) == "ActionGenerateFeedback"
        )
        if refinement_count >= self.prqe_config.max_refinement_rounds:
            logger.warning(
                f"[PRQE] 已达最大精炼轮数 ({self.prqe_config.max_refinement_rounds})，透传当前 PRD"
            )
            for msg in reversed(with_messages or []):
                if getattr(msg, "cause_by", None) == any_to_str(WritePRD):
                    return self._passthrough(msg)
            return None

        # 若 instruct_content 含 PrepareDocumentsOutput 结构（无 evaluation_result）则需先调用 WritePRD
        last_msg = with_messages[-1]
        ic = last_msg.instruct_content
        kvs = ic.model_dump() if ic and hasattr(ic, "model_dump") else (dict(ic) if ic else {})
        need_run_write_prd = (
            "requirements_filename" in kvs and "project_path" in kvs and "evaluation_result" not in kvs
        )

        write_prd_msg = None
        if need_run_write_prd:
            # 内部调用 WritePRD，避免未评估的 PRD 发布给 Architect
            write_prd = WritePRD(context=self.context, llm=self.llm)
            write_prd_msg = await write_prd.run(with_messages=with_messages, **kwargs)
            if not write_prd_msg:
                logger.warning("[PRQE] WritePRD 返回空")
                return None
        else:
            # 已有 WritePRD 输出（理论上 PRQE 模式下不会走此分支）
            for msg in reversed(with_messages):
                if getattr(msg, "cause_by", None) == any_to_str(WritePRD):
                    write_prd_msg = msg
                    break

        if not write_prd_msg or not write_prd_msg.instruct_content:
            logger.warning("[PRQE] 未找到 WritePRD 输出")
            return None

        ic = write_prd_msg.instruct_content
        kvs = ic.model_dump() if hasattr(ic, "model_dump") else dict(ic)
        changed_prd = kvs.get("changed_prd_filenames", [])
        project_path = kvs.get("project_path", "")

        if not changed_prd:
            changed_prd = kvs.get("prd_filenames", [])

        prd_content_parts = []
        for fp in changed_prd:
            doc = await Document.load(filename=fp, project_path=project_path)
            if doc and doc.content:
                prd_content_parts.append(doc.content)

        prd_content = "\n\n---\n\n".join(prd_content_parts) if prd_content_parts else write_prd_msg.content or ""

        if not prd_content:
            logger.warning("[PRQE] PRD 内容为空，透传原消息")
            return self._passthrough(write_prd_msg)

        evaluator = PRDQualityEvaluator(self.prqe_config)
        single_req = getattr(self.prqe_config, "single_requirement_mode", False)
        result = await evaluator.evaluate(prd_content, self.llm, single_requirement=single_req)

        if evaluator.is_passed(result):
            logger.info(f"[PRQE] 通过 (score={result.overall_score})，透传至 Architect")
            return self._passthrough(write_prd_msg)

        logger.info(f"[PRQE] 未通过 (score={result.overall_score})，触发精炼")
        return AIMessage(
            content=f"PRD 质量未达阈值 ({result.overall_score} < {self.prqe_config.threshold})",
            instruct_content=AIMessage.create_instruct_value(
                kvs={
                    "evaluation_result": result.model_dump(),
                    "project_path": project_path,
                    "requirements_filename": kvs.get("requirements_filename", ""),
                    "prd_filenames": kvs.get("prd_filenames", []),
                    "changed_prd_filenames": changed_prd,
                },
                class_name="PRQEEvaluationOutput",
            ),
            cause_by=any_to_str(ActionEvaluatePRD),
            send_to="Alice",
        )

    def _passthrough(self, write_prd_msg: Message) -> AIMessage:
        """透传 WritePRD 输出，确保 Architect 能收到。"""
        return AIMessage(
            content=write_prd_msg.content,
            instruct_content=write_prd_msg.instruct_content,
            cause_by=any_to_str(WritePRD),
            send_to="Bob",  # 仅发送给 Architect，避免 ProductManager 继续循环
        )
