#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove Chinese comments/docstrings from EGRCM/PRQE module files (keeps LLM prompts)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "metagpt/configs/prqe_config.py": '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PRQE (Product Requirement Quality Evaluator) configuration."""
from pydantic import BaseModel, Field


DEFAULT_WEIGHTS = {
    "completeness": 0.2,
    "clarity": 0.2,
    "feasibility": 0.2,
    "consistency": 0.15,
    "testability": 0.1,
    "traceability": 0.1,
    "atomicity": 0.05,
}


class PRQEConfig(BaseModel):
    """PRQE module configuration."""

    enabled: bool = False
    requirement_level: bool = True
    threshold: float = 8.0
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    max_refinement_rounds: int = 3
    single_requirement_mode: bool = False
''',
}


def patch_entropy_dialogue_engine(text: str) -> str:
    replacements = [
        ('# 原子级问题集默认路径（单条 SRS 条款用）\n', ''),
        ('    # 使用相对于 metagpt 包根目录的路径\n', ''),
        ('    # 是否使用真人问答模式；为 True 时由外部提供人类答案，False 时默认使用 LLM 自动澄清\n', ''),
        (
            '    # 需求类型：product=产品级需求（完整产品澄清），atomic=原子级/SRS 单条条款（仅做术语、约束、验收等澄清）\n',
            '',
        ),
        ('            # 相对路径时，相对于 metagpt 包根目录（即本文件上一级目录）\n', ''),
        ('        # 先按最小 phase，再按出现顺序\n', ''),
        (
            '        """原子级需求总结：仅基于本条与澄清信息，不添加无关产品或功能。"""\n',
            '',
        ),
        ('            # 低熵时只保证必答问题问完即可结束\n', ''),
        (
            '            # 原子级模式：始终以原始需求为语境，避免「产品名」主导后续回答\n'
            '            # 产品级模式：若已有产品名澄清，则将其与原始需求一并作为后续语境\n',
            '',
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def patch_module_docstring(text: str, new_doc: str) -> str:
    if text.startswith('#!/usr/bin/env python'):
        first = text.find('"""')
        if first == -1:
            return text
        second = text.find('"""', first + 3)
        if second == -1:
            return text
        return text[: first + 3] + new_doc + text[second + 3 :]
    return text


def patch_prd_quality_evaluator(text: str) -> str:
    text = patch_module_docstring(
        text,
        '\nPRQE core logic: 7-dimension PRD quality evaluation.\n',
    )
    replacements = [
        (
            'class PRQEEvaluationResult(BaseModel):\n'
            '    """PRQE 评估报告结构。\n\n'
            '    Attributes:\n'
            '        overall_score: 加权总分，范围 [0, 10]。\n'
            '        dimension_scores: 各维度分数 (0-10)。\n'
            '        issues: 发现的问题列表。\n'
            '        improvement_suggestions: 改进建议列表。\n'
            '        feedback_stage: 反馈路由阶段，由 FeedbackRouter 填充。\n'
            '    """\n',
            'class PRQEEvaluationResult(BaseModel):\n    """PRQE evaluation report."""\n',
        ),
        ('# 单需求文档专用评估 prompt：不强制要求目标用户、系统约束等产品级字段\n', ''),
        ('    """从文本中解析 JSON。"""\n', '    """Parse JSON from text."""\n'),
        ('    # 去掉 markdown 代码块\n', ''),
        ('    # 提取 {...}\n', ''),
        ('    """将分数限制在 0-10。"""\n', '    """Clamp score to 0-10."""\n'),
        (
            'class PRDQualityEvaluator:\n'
            '    """PRD 质量评估器。\n\n'
            '    根据 PRQEConfig 的权重计算加权总分，判断是否通过阈值。\n'
            '    """\n',
            'class PRDQualityEvaluator:\n    """PRD quality evaluator."""\n',
        ),
        ('        """计算加权总分。"""\n', '        """Compute weighted overall score."""\n'),
        (
            '        """解析 LLM 返回的 JSON 为 PRQEEvaluationResult。"""\n',
            '        """Parse LLM JSON into PRQEEvaluationResult."""\n',
        ),
        ('                issues=["解析失败"],\n', '                issues=["parse failed"],\n'),
        (
            '        """评估 PRD 质量。\n\n'
            '        Args:\n'
            '            prd_content: PRD 内容（JSON 或 Markdown 文本）。\n'
            '            llm: LLM 实例。\n'
            '            single_requirement: 若为 True，使用单需求专用 prompt。None 时从 config.single_requirement_mode 读取。\n\n'
            '        Returns:\n'
            '            PRQEEvaluationResult: 评估结果。\n'
            '        """\n',
            '        """Evaluate PRD quality."""\n',
        ),
        ('        """判断是否通过阈值。"""\n', '        """Return True if score meets threshold."""\n'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def patch_requirement_quality_evaluator(text: str) -> str:
    text = patch_module_docstring(
        text,
        '\nPRQE requirement-level quality control and refinement (before PRD generation).\n',
    )
    replacements = [
        ('    """判斷是否為 API 限流（429）錯誤。"""\n', '    """Return True if exception is a rate-limit (429) error."""\n'),
        ('# ============ 數據結構 ============\n\n', ''),
        ('    """單個需求問題。"""\n', '    """Single requirement issue."""\n'),
        ('    type: str = Field(description="問題類型")\n', '    type: str = Field(description="Issue type")\n'),
        (
            '    description: str = Field(description="問題描述")\n',
            '    description: str = Field(description="Issue description")\n',
        ),
        (
            'class RequirementEvaluationResult(BaseModel):\n'
            '    """需求質量評估結果。\n\n'
            '    Attributes:\n'
            '        scores: 各維度分數 (0-10)。\n'
            '        overall_score: 加權總分。\n'
            '        issues: 發現的問題列表（含 type 與 description）。\n'
            '        improvement_suggestions: 改進建議列表，可用於需求改寫。\n'
            '    """\n',
            'class RequirementEvaluationResult(BaseModel):\n    """Requirement quality evaluation result."""\n',
        ),
        ('# ============ 輔助函數 ============\n\n', ''),
        ('    """從文本中解析 JSON。"""\n', '    """Parse JSON from text."""\n'),
        ('    """將分數限制在 0-10。"""\n', '    """Clamp score to 0-10."""\n'),
        ('# ============ 評分 Prompt ============\n\n', ''),
        ('# ============ 問題檢測 Prompt ============\n\n', ''),
        ('# ============ 反饋生成 Prompt ============\n\n', ''),
        ('# ============ 需求優化 Prompt ============\n\n', ''),
        ('# ============ 類實現 ============\n\n', ''),
        (
            'class RequirementQualityEvaluator:\n'
            '    """需求質量評分器。\n\n'
            '    對單條需求或需求列表進行 7 維度評分。\n'
            '    """\n',
            'class RequirementQualityEvaluator:\n    """7-dimension requirement quality scorer."""\n',
        ),
        ('        """計算加權總分。"""\n', '        """Compute weighted overall score."""\n'),
        ('        """解析 LLM 返回的 JSON。"""\n', '        """Parse LLM JSON response."""\n'),
        (
            '        """評估需求質量。\n\n'
            '        Args:\n'
            '            requirement: 需求內容（單條或需求列表文本）。\n'
            '            llm: LLM 實例。\n\n'
            '        Returns:\n'
            '            RequirementEvaluationResult: 評估結果。\n'
            '        """\n',
            '        """Evaluate requirement quality."""\n',
        ),
        ('        """判斷是否通過閾值。"""\n', '        """Return True if score meets threshold."""\n'),
        (
            'class RequirementIssueDetector:\n'
            '    """需求問題檢測器。\n\n'
            '    檢測 missing_requirement_elements、ambiguous_descriptions、\n'
            '    missing_measurable_constraints、missing_acceptance_criteria、\n'
            '    missing_requirement_id、conflicting_requirements 等問題。\n'
            '    """\n',
            'class RequirementIssueDetector:\n    """Detect requirement quality issues."""\n',
        ),
        ('        """解析 LLM 返回的問題列表。"""\n', '        """Parse issue list from LLM response."""\n'),
        (
            '        """檢測需求問題。\n\n'
            '        Args:\n'
            '            requirement: 需求內容。\n'
            '            llm: LLM 實例。\n\n'
            '        Returns:\n'
            '            問題列表。\n'
            '        """\n',
            '        """Detect issues in a requirement."""\n',
        ),
        (
            'class RequirementFeedbackGenerator:\n'
            '    """需求改進建議生成器。\n\n'
            '    根據評估結果與問題列表，生成可用於需求改寫的結構化反饋。\n'
            '    """\n',
            'class RequirementFeedbackGenerator:\n    """Generate structured refinement feedback."""\n',
        ),
        ('        """解析 LLM 返回的改進建議。"""\n', '        """Parse improvement suggestions from LLM response."""\n'),
        (
            '        """生成改進建議。\n\n'
            '        Args:\n'
            '            requirement: 需求內容。\n'
            '            evaluation_result: 評估結果（含 scores 與 issues）。\n'
            '            llm: LLM 實例。\n\n'
            '        Returns:\n'
            '            改進建議列表。\n'
            '        """\n',
            '        """Generate improvement suggestions."""\n',
        ),
        (
            'class RequirementRefiner:\n'
            '    """需求優化器。\n\n'
            '    根據 PRQE 反饋生成新的需求。\n'
            '    """\n',
            'class RequirementRefiner:\n    """Refine requirements from PRQE feedback."""\n',
        ),
        (
            '        """根據反饋改寫需求。\n\n'
            '        Args:\n'
            '            requirement: 原始需求。\n'
            '            feedback: 改進建議列表，或 RequirementEvaluationResult（將使用其 improvement_suggestions）。\n'
            '            llm: LLM 實例。\n\n'
            '        Returns:\n'
            '            優化後的需求文本。\n'
            '        """\n',
            '        """Refine requirement from feedback."""\n',
        ),
        ('# ============ 統一接口 ============\n\n', ''),
        (
            '    """根據 PRQE 反饋生成新的需求。\n\n'
            '    便捷函數，內部調用 RequirementRefiner.refine。\n\n'
            '    Args:\n'
            '        requirement: 原始需求。\n'
            '        feedback: 改進建議列表或 RequirementEvaluationResult。\n'
            '        llm: LLM 實例。\n\n'
            '    Returns:\n'
            '        優化後的需求文本。\n'
            '    """\n',
            '    """Refine requirement; convenience wrapper around RequirementRefiner.refine."""\n',
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def patch_code_adapter(text: str) -> str:
    return patch_module_docstring(
        text,
        '\nCode Adapter - convert structured PRQE output to code-oriented task descriptions.\n',
    ).replace(
        '僅在代碼生成任務時使用，不參與 PRD 評估實驗。\n'
        '將產品級、用戶級描述轉換為結構化編程格式：Task, Input, Output, Constraints。\n'
        '保留函數簽名與 >>> 示例，移除產品級/用戶級描述。\n',
        'Used for code generation tasks only.\n',
    ).replace('@Time    : 2025/3/17\n@Desc    : Code Adapter - 將 PRQE 結構化輸出轉換為代碼導向的任務描述。\n\n', '')


def patch_rpc_tool(text: str) -> str:
    return patch_module_docstring(
        text,
        '\nRPC (Requirement-to-Prompt Compression) - compress structured requirements for code generation.\n',
    ).replace('僅在代碼生成任務時使用，不參與 PRD 評估實驗。\n', 'Used for code generation tasks only.\n').replace(
        '@Time    : 2025/3/16\n@Desc    : RPC (Requirement-to-Prompt Compression) - 將結構化需求壓縮為代碼生成 prompt。\n\n',
        '',
    )


def patch_pipeline_controller(text: str) -> str:
    return patch_module_docstring(
        text,
        '''
Pipeline Controller - route tasks to PRD or code generation after PRQE.

Flow:
  Requirement -> EGRCM -> PRQE -> Task Router
  |-- PRD Flow    -> PRD Generation
  `-- Code Flow   -> Code Adapter -> RPC -> Code Generation

Experiment 2 bypasses this controller (Requirement -> EGRCM -> PRQE -> PRD).
Adapter and RPC are used only for code-generation tasks.
''',
    ).replace(
        '@Time    : 2025/3/16\n'
        '@Desc    : Pipeline Controller - 統一管線控制，根據任務類型路由到 PRD 或代碼生成。\n\n'
        '流程：\n'
        '  Requirement → EGRCM → PRQE → Task Router\n'
        '  ├── PRD Flow (unchanged)    → PRD Generation\n'
        '  └── Code Flow (new)         → Code Adapter → RPC → Code Generation\n\n'
        '重要：PRD 評估實驗（Experiment 2）不經過此控制器，直接使用 Requirement → EGRCM → PRQE → PRD。\n'
        'Adapter 與 RPC 僅在代碼生成任務時調用。\n',
        '',
    )


def patch_cp_egrcm(text: str) -> str:
    return patch_module_docstring(
        text,
        '''
CP-EGRCM (Contract-Preserving EGRCM) - contract-preserving refinement for code tasks.

Used for code generation only. Preserves signature, return type, and examples.
''',
    ).replace(
        '@Time    : 2025/3/17\n'
        '@Desc    : CP-EGRCM (Contract-Preserving EGRCM) - 契約保持型需求精煉。\n\n'
        '僅用於代碼生成任務，不修改現有 EGRCM。\n'
        '保持函數簽名、返回類型、示例不變，僅澄清語義。\n',
        '',
    )


def patch_feedback_router(text: str) -> str:
    return patch_module_docstring(text, '\nMap PRQE issue types to entropy dialogue stages.\n').replace(
        '@Time    : 2025/3/9\n@Desc    : 将 PRQE 问题类型映射到熵对话阶段。\n',
        '',
    ).replace('# 问题类型关键词 -> 路由阶段\n# 对应 requirement_questions.yaml 的 phase\n', '').replace(
        '    """根据问题与建议将反馈路由到熵对话阶段。\n\n'
        '    映射规则：\n'
        '    - Missing Requirements / Feature Scope Too Large -> feature_decomposition (phase 2)\n'
        '    - Ambiguous Requirement -> requirement_quantification (phase 1)\n'
        '    - Requirement Conflict -> product_boundary (phase 3)\n'
        '    - Missing Priority -> priority_confirmation (phase 4)\n\n'
        '    Args:\n'
        '        issues: 问题列表。\n'
        '        suggestions: 改进建议列表。\n\n'
        '    Returns:\n'
        '        反馈阶段标识，如 "feature_decomposition"。\n'
        '    """\n',
        '    """Route feedback to an entropy dialogue stage."""\n',
    ).replace('        return "feature_decomposition"  # 默认\n', '        return "feature_decomposition"\n')


def main() -> None:
    patch_fns = {
        "metagpt/modules/entropy_dialogue_engine.py": patch_entropy_dialogue_engine,
        "metagpt/modules/gating.py": lambda t: t,
        "metagpt/modules/pipeline_controller.py": patch_pipeline_controller,
        "metagpt/modules/code_adapter.py": patch_code_adapter,
        "metagpt/modules/rpc_tool.py": patch_rpc_tool,
        "metagpt/modules/requirement_quality_evaluator.py": patch_requirement_quality_evaluator,
        "metagpt/modules/prd_quality_evaluator.py": patch_prd_quality_evaluator,
        "metagpt/modules/feedback_router.py": patch_feedback_router,
        "metagpt/modules/cp_egrcm.py": patch_cp_egrcm,
    }

    for rel, content in FILES.items():
        path = ROOT / rel
        path.write_text(content, encoding="utf-8")
        print(f"updated: {rel}")

    for rel, fn in patch_fns.items():
        path = ROOT / rel
        if not path.exists():
            raise SystemExit(f"missing: {path}")
        original = path.read_text(encoding="utf-8")
        updated = fn(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            print(f"updated: {rel}")
        else:
            print(f"unchanged: {rel}")


if __name__ == "__main__":
    main()
