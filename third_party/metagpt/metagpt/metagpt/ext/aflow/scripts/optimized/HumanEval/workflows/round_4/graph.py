from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HumanEval.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HumanEval.workflows.round_4.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.llm.cost_manager = CostManager()
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.test = operator.Test(self.llm)
        self.review = operator.Review(self.llm)  # Added Review operator

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        # Generate multiple solutions using custom_code_generate
        solutions = []
        for _ in range(3):  # Generate three solutions for ensemble
            response = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction="")
            solutions.append(response['response'])
        
        # Use ScEnsemble to select the best solution
        best_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        # Review the best solution before testing
        review_result = await self.review(solution=best_solution['response'])
        
        # Test the best solution if it passes review
        if review_result['valid']:
            test_result = await self.test(problem=problem, solution=best_solution['response'], entry_point=entry_point)
            if test_result['result']:
                return best_solution['response'], self.llm.cost_manager.total_cost
            else:
                return "No valid solution found after testing.", self.llm.cost_manager.total_cost
        else:
            return "Solution failed review.", self.llm.cost_manager.total_cost
