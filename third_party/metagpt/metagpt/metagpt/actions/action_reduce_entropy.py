from __future__ import annotations

from pydantic import Field

from metagpt.actions import Action
from metagpt.actions.add_requirement import UserRequirement
from metagpt.logs import logger
from metagpt.modules.entropy_dialogue_engine import (
    EntropyDialogueConfig,
    EntropyReductionDialogueEngine,
)
from metagpt.schema import AIMessage
from metagpt.states.requirement_state import RequirementState
from metagpt.utils.common import any_to_str
from metagpt.utils.human_interaction import HumanInteraction


class ActionReduceEntropy(Action):
    """根据用户原始需求进行多轮澄清，生成低熵需求描述。"""

    name: str = "ActionReduceEntropy"
    # 与 PrepareDocuments 保持一致，支持在 set_actions 中通过 send_to 路由消息
    send_to: str = ""
    # 使用单独的 entropy_config 字段，避免与基类 Action.config 混淆
    entropy_config: EntropyDialogueConfig | None = Field(default=None, exclude=True)
    engine: EntropyReductionDialogueEngine | None = Field(default=None, exclude=True)

    async def run(self, with_messages, **kwargs):
        # 从最近的消息中找到最新的 UserRequirement
        req_msg = None
        for msg in reversed(with_messages or []):
            if msg.cause_by == any_to_str(UserRequirement):
                req_msg = msg
                break

        if not req_msg:
            logger.warning("No UserRequirement message found for ActionReduceEntropy")
            return None

        state = RequirementState(raw_requirement=req_msg.content)

        if not self.engine:
            self.engine = EntropyReductionDialogueEngine(self.entropy_config or EntropyDialogueConfig())

        use_human = bool(self.engine.config.use_human)

        if use_human:
            # 真人问答模式：一行输入 + 回车即可结束当前问题

            async def ask_fn(question_text: str) -> str:
                logger.info(f"[Entropy QA][Human] 问题：{question_text}")
                prompt = (
                    f"\n【需求澄清问题】{question_text}\n"
                    f"请使用简体中文回答，直接按回车键结束当前问题的输入："
                    "\n> "
                )
                # 这里使用同步 input，行为与其他交互式工具保持一致
                return input(prompt)

        else:

            async def ask_fn(question_text: str) -> str:
                logger.info(f"[Entropy QA][LLM] Q: {question_text}")
                return await self.llm.aask(question_text)

        refined_state = await self.engine.refine(
            state=state,
            ask_fn=ask_fn,
            llm=self.llm,
        )

        # 输出结构化需求，方便下游或调试使用
        return AIMessage(
            content=refined_state.refined_requirement or refined_state.raw_requirement,
            instruct_content=refined_state.model_dump(),
            cause_by=any_to_str(ActionReduceEntropy),
            send_to=self.send_to,
        )

