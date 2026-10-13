# -*- coding: utf-8 -*-
"""Accumulate OpenAI chat.completions usage across experiment runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ProblemUsage:
    task_id: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0


@dataclass
class TokenTracker:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    problems: list[ProblemUsage] = field(default_factory=list)
    call_log: list[dict] = field(default_factory=list)
    problem_audits: dict = field(default_factory=dict)
    _cur: Optional[ProblemUsage] = field(default=None, repr=False)

    def start_problem(self, task_id: str) -> None:
        self._cur = ProblemUsage(task_id=task_id)

    def end_problem(self) -> None:
        if self._cur is not None:
            self.problems.append(self._cur)
            self._cur = None

    def add_usage(self, usage: Any, meta: Optional[dict] = None) -> None:
        if usage is None:
            return
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        tt = int(getattr(usage, "total_tokens", 0) or (pt + ct))
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_tokens += tt
        self.llm_calls += 1
        if self._cur is not None:
            self._cur.prompt_tokens += pt
            self._cur.completion_tokens += ct
            self._cur.total_tokens += tt
            self._cur.llm_calls += 1
        if meta is not None:
            task_id = None
            if self._cur is not None:
                task_id = self._cur.task_id
            task_id = meta.get("task_id") or task_id
            rec = {
                "task_id": task_id,
                "stage": meta.get("stage", "unknown"),
                "branch": meta.get("branch", ""),
                "sample_index": meta.get("sample_index"),
                "issue": meta.get("issue", ""),
                "reason": meta.get("reason", ""),
                "drivers": list(meta.get("drivers") or []),
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
            }
            for key in (
                "question_id",
                "temperature",
                "prompt_chars",
                "system_chars",
                "user_chars",
                "extra_context_chars",
                "prqe_applied",
                "failure",
            ):
                if key in meta:
                    rec[key] = meta[key]
            self.call_log.append(rec)

    def summary(self, *, model: str, method: str, dataset: str, n_problems: int) -> dict:
        avg = (self.total_tokens / n_problems) if n_problems else 0.0
        # gpt-4o-mini list price (USD / 1M)
        cost = self.prompt_tokens * 0.15 / 1e6 + self.completion_tokens * 0.60 / 1e6
        return {
            "method": method,
            "model": model,
            "dataset": dataset,
            "n_problems": n_problems,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
            "avg_tokens_per_problem": round(avg, 2),
            "avg_llm_calls_per_problem": round(self.llm_calls / n_problems, 3) if n_problems else 0.0,
            "est_cost_usd": round(cost, 6),
            "pricing_usd_per_million": {"input": 0.15, "output": 0.60},
            "note": "Recorded from live response.usage during generation.",
            "problems": [asdict(p) for p in self.problems],
        }


# Module-level tracker used by experiment scripts
TRACKER = TokenTracker()
