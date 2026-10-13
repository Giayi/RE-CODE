#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2025/3/9
@Desc    : PRQE 反馈生成：分析评估报告，更新需求文件，触发 PRD 重新生成。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from metagpt.actions import Action
from metagpt.actions.action_evaluate_prd import ActionEvaluatePRD
from metagpt.actions.prepare_documents import PrepareDocuments
from metagpt.logs import logger
from metagpt.modules.feedback_router import route
from metagpt.modules.prd_quality_evaluator import PRQEEvaluationResult
from metagpt.schema import AIMessage, Message
from metagpt.utils.common import any_to_str, aread, awrite


class ActionGenerateFeedback(Action):
    """根据 PRQE 评估结果生成改进指令，更新需求文件，触发 WritePRD 重新执行。"""

    name: str = "ActionGenerateFeedback"

    async def run(
        self,
        with_messages: Optional[List[Message]] = None,
        **kwargs,
    ) -> Optional[AIMessage]:
        """分析评估报告，更新需求文件，输出可触发 WritePRD 的消息。"""
        if not with_messages:
            logger.warning("[PRQE Feedback] 无输入消息")
            return None

        # 查找 ActionEvaluatePRD 输出
        eval_msg = None
        for msg in reversed(with_messages):
            if getattr(msg, "cause_by", None) == any_to_str(ActionEvaluatePRD):
                eval_msg = msg
                break

        if not eval_msg or not eval_msg.instruct_content:
            logger.warning("[PRQE Feedback] 未找到 ActionEvaluatePRD 输出")
            return None

        ic = eval_msg.instruct_content
        kvs = ic.model_dump() if hasattr(ic, "model_dump") else dict(ic)
        eval_result = kvs.get("evaluation_result", {})
        project_path = kvs.get("project_path", "")
        requirements_filename = kvs.get("requirements_filename", "")
        prd_filenames = kvs.get("prd_filenames", [])

        if not requirements_filename or not Path(requirements_filename).exists():
            logger.warning(f"[PRQE Feedback] 需求文件不存在: {requirements_filename}")
            return None

        # 路由反馈阶段
        issues = eval_result.get("issues", [])
        suggestions = eval_result.get("improvement_suggestions", [])
        feedback_stage = route(issues, suggestions)
        eval_result["feedback_stage"] = feedback_stage

        # 读取当前需求，追加 PRQE 反馈
        try:
            current_req = await aread(requirements_filename)
        except Exception as e:
            logger.error(f"[PRQE Feedback] 读取需求文件失败: {e}")
            return None

        feedback_block = "\n\n## PRQE 改进建议（请据此完善需求后重新生成 PRD）\n"
        feedback_block += f"- 反馈阶段: {feedback_stage}\n"
        if issues:
            feedback_block += "- 问题:\n"
            for i in issues:
                feedback_block += f"  - {i}\n"
        if suggestions:
            feedback_block += "- 改进建议:\n"
            for s in suggestions:
                feedback_block += f"  - {s}\n"

        new_content = (current_req or "").rstrip() + feedback_block

        try:
            await awrite(requirements_filename, new_content)
        except Exception as e:
            logger.error(f"[PRQE Feedback] 写入需求文件失败: {e}")
            return None

        logger.info(f"[PRQE Feedback] 已更新需求文件，反馈阶段: {feedback_stage}")

        # 输出与 PrepareDocumentsOutput 相同结构，供 WritePRD 重新执行
        return AIMessage(
            content="PRQE 反馈已写入需求文件，请重新生成 PRD。",
            instruct_content=AIMessage.create_instruct_value(
                kvs={
                    "project_path": project_path,
                    "requirements_filename": requirements_filename,
                    "prd_filenames": prd_filenames,
                },
                class_name="PrepareDocumentsOutput",
            ),
            cause_by=any_to_str(ActionGenerateFeedback),
            send_to="Alice",
        )
