#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : product_manager.py
@Modified By: liushaojie, 2024/10/17.
@Modified By: PRQE integration, 2025/3/9.
"""
from metagpt.actions import UserRequirement, WritePRD
from metagpt.actions.action_evaluate_prd import ActionEvaluatePRD
from metagpt.actions.action_evaluate_requirement import (
    ActionEvaluateRequirement,
    ActionRefineRequirement,
)
from metagpt.actions.action_generate_feedback import ActionGenerateFeedback
from metagpt.actions.action_reduce_entropy import ActionReduceEntropy
from metagpt.actions.prepare_documents import PrepareDocuments
from metagpt.actions.search_enhanced_qa import SearchEnhancedQA
from metagpt.configs.prqe_config import PRQEConfig
from metagpt.modules.entropy_dialogue_engine import EntropyDialogueConfig
from metagpt.prompts.product_manager import PRODUCT_MANAGER_INSTRUCTION
from metagpt.roles.di.role_zero import RoleZero
from metagpt.roles.role import RoleReactMode
from metagpt.tools.libs.browser import Browser
from metagpt.tools.libs.editor import Editor
from metagpt.utils.common import any_to_name, any_to_str, tool2name
from metagpt.utils.git_repository import GitRepository


class ProductManager(RoleZero):
    """
    Represents a Product Manager role responsible for product development and management.

    Attributes:
        name (str): Name of the product manager.
        profile (str): Role profile, default is 'Product Manager'.
        goal (str): Goal of the product manager.
        constraints (str): Constraints or limitations for the product manager.
    """

    name: str = "Alice"
    profile: str = "Product Manager"
    goal: str = "Create a Product Requirement Document or market research/competitive product research."
    constraints: str = "utilize the same language as the user requirements for seamless communication"
    instruction: str = PRODUCT_MANAGER_INSTRUCTION
    tools: list[str] = ["RoleZero", Browser.__name__, Editor.__name__, SearchEnhancedQA.__name__]

    todo_action: str = any_to_name(WritePRD)
    entropy_config: EntropyDialogueConfig | None = None
    prqe_config: PRQEConfig | None = None

    @property
    def use_entropy_dialogue(self) -> bool:
        return bool(self.entropy_config and self.entropy_config.enabled)

    @property
    def use_prqe(self) -> bool:
        return bool(self.prqe_config and self.prqe_config.enabled)

    @property
    def use_prqe_requirement_level(self) -> bool:
        """是否使用需求級 PRQE（在 PRD 生成之前評估需求）。"""
        return bool(
            self.prqe_config
            and self.prqe_config.enabled
            and getattr(self.prqe_config, "requirement_level", True)
        )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.use_fixed_sop:
            self.enable_memory = False
            if self.use_entropy_dialogue:
                base_actions = [
                    ActionReduceEntropy(entropy_config=self.entropy_config, send_to=any_to_str(self)),
                    PrepareDocuments(send_to=any_to_str(self)),
                ]
            else:
                base_actions = [PrepareDocuments(send_to=any_to_str(self))]

            if self.use_prqe:
                if self.use_prqe_requirement_level:
                    # 新流程：需求級 PRQE，在 PRD 生成之前評估與精煉需求
                    if self.use_entropy_dialogue:
                        base_actions = [
                            ActionReduceEntropy(entropy_config=self.entropy_config, send_to=any_to_str(self)),
                            ActionEvaluateRequirement(prqe_config=self.prqe_config),
                            ActionRefineRequirement(prqe_config=self.prqe_config),
                            PrepareDocuments(send_to=any_to_str(self)),
                            WritePRD,
                        ]
                    else:
                        base_actions = [
                            ActionEvaluateRequirement(prqe_config=self.prqe_config),
                            ActionRefineRequirement(prqe_config=self.prqe_config),
                            PrepareDocuments(send_to=any_to_str(self)),
                            WritePRD,
                        ]
                    watch_list = [UserRequirement, ActionEvaluateRequirement, ActionRefineRequirement, PrepareDocuments]
                    if self.use_entropy_dialogue:
                        watch_list = [UserRequirement, ActionReduceEntropy, ActionEvaluateRequirement, ActionRefineRequirement, PrepareDocuments]
                    self.set_actions(base_actions)
                    self._watch(watch_list)
                else:
                    # 舊流程：PRD 級 PRQE
                    base_actions.extend([
                        ActionEvaluatePRD(prqe_config=self.prqe_config),
                        ActionGenerateFeedback,
                    ])
                    watch_list = [UserRequirement, PrepareDocuments, ActionEvaluatePRD, ActionGenerateFeedback]
                    if self.use_entropy_dialogue:
                        watch_list = [UserRequirement, ActionReduceEntropy, PrepareDocuments, ActionEvaluatePRD, ActionGenerateFeedback]
                    self.set_actions(base_actions)
                    self._watch(watch_list)
            else:
                base_actions.append(WritePRD)
                self.set_actions(base_actions)
                if self.use_entropy_dialogue:
                    self._watch([UserRequirement, ActionReduceEntropy, PrepareDocuments])
                else:
                    self._watch([UserRequirement, PrepareDocuments])
            self.rc.react_mode = RoleReactMode.BY_ORDER

    def _update_tool_execution(self):
        wp = WritePRD()
        self.tool_execution_map.update(tool2name(WritePRD, ["run"], wp.run))

    async def _think(self) -> bool:
        """Decide what to do"""
        if not self.use_fixed_sop:
            return await super()._think()

        if GitRepository.is_git_dir(self.config.project_path) and not self.config.git_reinit:
            self._set_state(1)
        else:
            self._set_state(0)
            self.config.git_reinit = False
            self.todo_action = any_to_name(WritePRD)
        return bool(self.rc.todo)
