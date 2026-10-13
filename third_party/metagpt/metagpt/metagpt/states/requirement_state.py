from typing import List, Optional

from pydantic import BaseModel, Field


class Clarification(BaseModel):
    question_id: str
    question: str
    answer: str
    dimension: str
    phase: int


class RequirementState(BaseModel):
    raw_requirement: str
    clarifications: List[Clarification] = Field(default_factory=list)
    refined_requirement: Optional[str] = None
    # 1.0 表示高度不确定，0 表示需求已经非常清晰
    entropy_score: float = 1.0

    def is_low_entropy(self, threshold: float = 0.3) -> bool:
        return self.entropy_score <= threshold

    def update_from_qa(self, question_obj, answer: str) -> None:
        self.clarifications.append(
            Clarification(
                question_id=question_obj.id,
                question=question_obj.prompt,
                answer=answer,
                dimension=question_obj.dimension,
                phase=question_obj.phase,
            )
        )
        # 简单熵衰减策略：每次澄清将熵乘以一个系数
        self.entropy_score = max(0.0, self.entropy_score * 0.7)

