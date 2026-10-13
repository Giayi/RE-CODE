from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable, List, Literal, Optional

import yaml
from pydantic import BaseModel

from metagpt.states.requirement_state import RequirementState

DEFAULT_ATOMIC_QUESTIONS_FILE = "prompts/requirement_questions_atomic.yaml"


class EntropyDialogueConfig(BaseModel):
    enabled: bool = False
    questions_file: str = "prompts/requirement_questions.yaml"
    entropy_threshold: float = 0.3
    max_rounds: int = 5
    use_human: bool = False
    requirement_type: Literal["product", "atomic"] = "product"


class RequirementQuestion(BaseModel):
    id: str
    phase: int
    dimension: str
    prompt: str
    optional: bool = False


class EntropyReductionDialogueEngine:
    def __init__(self, config: EntropyDialogueConfig):
        self.config = config
        self.questions: Optional[List[RequirementQuestion]] = None

    def _load_questions_from_yaml(self, path: str) -> List[RequirementQuestion]:
        file_path = Path(path)
        if not file_path.is_absolute():
            package_root = Path(__file__).resolve().parents[1]
            file_path = package_root / path

        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        return [RequirementQuestion(**q) for q in data]

    def _select_next_question(self, state: RequirementState) -> Optional[RequirementQuestion]:
        if not self.questions:
            return None

        asked_ids = {c.question_id for c in state.clarifications}
        remaining = [q for q in self.questions if q.id not in asked_ids]
        if not remaining:
            return None

        min_phase = min(q.phase for q in remaining)
        same_phase = [q for q in remaining if q.phase == min_phase]
        return same_phase[0] if same_phase else None

    def _simple_sum(self, state: RequirementState) -> str:
        lines: List[str] = [f"原始需求：{state.raw_requirement}", "", "澄清要点："]
        if not state.clarifications:
            lines.append("- （暂无澄清，直接使用原始需求）")
        else:
            for c in state.clarifications:
                lines.append(f"- ({c.dimension}) {c.question} → {c.answer}")
        return "\n".join(lines)

    def _summarize(self, state: RequirementState) -> str:
        clar_text = "\n".join(f"{c.dimension}: {c.answer}" for c in state.clarifications)
        return f"""
请基于以下信息生成一个结构化且明确的最终需求描述：

原始需求：
{state.raw_requirement}

澄清信息：
{clar_text}

请使用简体中文，以简洁、可执行的需求描述方式总结。
"""

    def _summarize_atomic(self, state: RequirementState) -> str:
        clar_text = "\n".join(f"{c.dimension}: {c.answer}" for c in state.clarifications)
        return f"""
请仅基于下面这一条原始需求及其澄清信息，生成一条结构化、可验证的最终需求描述。
约束：不得发明与本条无关的产品名称、用户画像或功能；仅提炼与本条直接相关的要求、依赖、验收条件与术语说明。

原始需求：
{state.raw_requirement}

澄清信息：
{clar_text}

请使用简体中文，输出一条简洁、可执行的需求描述。若澄清信息中包含目标用户、系统约束，请在输出中体现；格式可分段：需求陈述 / 目标用户（若有）/ 依赖与约束 / 验收条件。
"""

    def _is_atomic_mode(self) -> bool:
        return self.config.requirement_type == "atomic"

    async def refine(
        self,
        *,
        state: RequirementState,
        ask_fn: Callable[[str], Awaitable[str]],
        llm=None,
    ) -> RequirementState:
        if not self.questions:
            path = (
                DEFAULT_ATOMIC_QUESTIONS_FILE
                if self._is_atomic_mode()
                else self.config.questions_file
            )
            self.questions = self._load_questions_from_yaml(path)

        rounds = 0
        while rounds < self.config.max_rounds:
            if state.is_low_entropy(self.config.entropy_threshold):
                remaining_must = [
                    q
                    for q in self.questions
                    if not q.optional and q.id not in {c.question_id for c in state.clarifications}
                ]
                if not remaining_must:
                    break

            q = self._select_next_question(state)
            if not q:
                break

            context_requirement = state.raw_requirement
            if not self._is_atomic_mode():
                product_clar = next(
                    (
                        c
                        for c in state.clarifications
                        if c.question_id == "product_name" or c.dimension == "product"
                    ),
                    None,
                )
                if product_clar:
                    context_requirement = f"{product_clar.answer}（原始需求：{state.raw_requirement}）"

            question_text = q.prompt.format(requirement=context_requirement)
            answer = await ask_fn(question_text)

            state.update_from_qa(q, answer)
            rounds += 1

        if llm:
            summary_prompt = (
                self._summarize_atomic(state)
                if self._is_atomic_mode()
                else self._summarize(state)
            )
            state.refined_requirement = await llm.aask(summary_prompt)
        else:
            state.refined_requirement = self._simple_sum(state)

        return state

