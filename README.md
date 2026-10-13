# RE-CODE

**RE-CODE** (Requirement Engineering for CODE generation) clarifies and refines requirements before code generation using entropy-guided dialogue (**EGRCM**) and requirement quality evaluation (**PRQE**). For programming tasks it further applies a Code Adapter and **RPC** to compress prompts, and is evaluated on HumanEval, MBPP, and other benchmarks against multiple baselines.

This repository is the **code + public experiment artifacts** bundle accompanying the paper (primarily **GPT-4o-mini**).

---

## Repository layout

```
RE-CODE/
├── README.md                 # This file
├── code/recode/              # RE-CODE core modules and experiment scripts
│   ├── modules/              # EGRCM, PRQE, router, RPC, etc.
│   ├── scripts/              # Experiments 1–3, repeat runs, evaluation
│   └── result_fig/           # Scripts for paper figures
├── data/
│   ├── datasets/             # Benchmark and requirement subsets
│   ├── baselines/            # Baseline method outputs
│   └── recode/               # RE-CODE method outputs
├── third_party/metagpt/      # MetaGPT fork used for integrated experiments
└── docs/
    └── REPRODUCTION.md       # Reproduction guide and legacy path mapping
```

See [`data/README.md`](data/README.md) for a full data index.

---

## Method overview

| Component | Role |
|-----------|------|
| **EGRCM** | Multi-turn requirement clarification via entropy and gating (`modules/cp_egrcm.py`, `entropy_dialogue_engine.py`) |
| **PRQE** | Requirement quality assessment and iterative refinement (`requirement_quality_evaluator.py`) |
| **Task Router + Pipeline** | After PRQE, route to PRD vs. code generation (`pipeline_controller.py`) |
| **Code Adapter + RPC** | Preserve signatures/examples and compress prompts for code tasks (`code_adapter.py`, `rpc_tool.py`) |

In code-generation experiments, the full pipeline is named **`metagpt_egrcm_prqe`** (i.e., **RE-CODE** in the paper).

---

## Experiments and data locations

| Paper experiment | Data location |
|------------------|---------------|
| Requirement quality / entropy (Exp. 1) | `data/datasets/pure`, `promise`, `frnfr`; scripts in `code/recode/scripts/experiment1/` |
| PRQE refinement (Exp. 2) | `code/recode/scripts/experiment2/` |
| Code-generation ablation (Exp. 3) | `data/recode/experiment3_ablation/` |
| RE-CODE, 5 repeat runs | `data/recode/repeat_egrcm_prqe/summary.md` |
| MetaGPT baseline, 5 repeats | `data/baselines/metagpt_repeat/summary.md` |
| Direct LLM | `data/baselines/direct_llm/` |
| Reflexion | `data/baselines/reflexion/repeat/summary/summary.md` |
| ClarifyGPT | `data/baselines/clarifygpt_repeat/report.md` |
| OpenHands | `data/baselines/openhands/repeat_openhands/aggregate/REPORT.md` |
| Token breakdown by stage | `data/recode/token_analysis/stage_token_analysis.md` |

---

## Quick start

### Read main results

- RE-CODE repeat runs: `data/recode/repeat_egrcm_prqe/summary.md`
- Baseline summaries: `data/baselines/*/summary.md`, or `report.md` / `REPORT.md` where applicable

### Reproduce (requires API access and dependencies)

1. Configure your LLM API (do not commit secrets).
2. Install or mount `third_party/metagpt` and `code/recode` according to your local MetaGPT integration.
3. Follow [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md); use dataset paths under `data/datasets/`.

### Figures

```bash
cd code/recode/result_fig
python experiment3_gpt4o_mini.py
```

---

## Citation and licensing

Please cite this work when publishing. Third-party frameworks (MetaGPT, Reflexion, ClarifyGPT, OpenHands, etc.) have their own licenses and citation requirements.

`third_party/metagpt` is a research integration copy and is not the official upstream release.

---

## Security

The public artifact bundle **does not** include API keys. If local notes contained secrets before packaging, rotate those keys and confirm they are not in the repository before release.

---

## Contact

Author information and issue contact details can be added here after publication.
