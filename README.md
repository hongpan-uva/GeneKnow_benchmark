# GeneKnow Benchmark Evaluation

Supporting material for the GeneKnow benchmark.
This repository contains workflows, scripts, test cases, and evaluation results for benchmarking GeneKnow against other methods on biomedical literature retrieval and claim verification tasks.

## Repository Structure

| Directory         | Description                                                            |
|-------------------|------------------------------------------------------------------------|
| `workflow/`       | Step-by-step evaluation protocols                                      |
| `scripts/`        | Python/R scripts for automated extraction, verification, and evaluation |
| `discover_cases/` | Test cases and results for the discover benchmark                      |
| `inspect_cases/`  | Test cases and results for the inspect benchmark                       |

## Data Availability

- All raw method outputs and evaluation results are included in this repository under `discover_cases/` and `inspect_cases/`.
- Full-text reference papers are **not** included for copyright and licensing reasons; reviewers should obtain them independently via the provided DOIs/PMIDs.

## Workflows

- [Discover Benchmark Workflow](workflow/discover_evaluation.md) — Open-domain retrieval and synopsis generation.
- [Inspect Benchmark Workflow](workflow/inspect_evaluation.md) — Targeted paper inspection and alignment evaluation.

## Quick Start

1. Set the `OPENAI_API_KEY` environment variable.
2. Set `SCRIPTS` and `ABLATION` to the absolute paths of `scripts/eval/` and `scripts/ablation/` in this repository.
3. Follow the instructions in the workflow documents under `workflow/`.

## Dependencies

- Python 3.x
- Install required packages: `pip install -r requirements.txt`
- The ablation baseline scripts (`naive_inspect_1.py`, `naive_inspect_2.py`) require the `geneknow` Python package.
