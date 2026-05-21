# Inspect Benchmark Workflow

Targeted paper inspection and alignment evaluation.
This workflow evaluates how well a method inspects a single target paper (identified by PMID) and extracts claims aligned with a gene and context.

## Methods

- **GeneKnow_inspect** — The GeneKnow inspection pipeline.
- **naive_1** (*Full-paper LLM*) — Feeds the entire full-text of the target paper directly to an LLM to generate a gene-context summary, without passage retrieval or verification.
- **naive_2** (*Hierarchical LLM*) — Retrieves relevant evidence passages from the paper, summarizes each passage individually, and then synthesizes a final article summary from those passage summaries; skips verification.

## Prerequisites

- Python 3.x
- `geneknow` CLI installed and on `PATH`
- `OPENAI_API_KEY` environment variable set
- Set working variables for the evaluation code directory and the GeneKnow scripts directory:
  ```bash
  SCRIPTS=/path/to/evaluation/scripts/eval
  ABLATION=/path/to/evaluation/scripts/ablation
  ```

## Test Cases

| id | gene       | context                                                   | PMID     |
|----|------------|-----------------------------------------------------------|----------|
| 1  | SOX9       | astrocyte astrocytic                                      | 41271638 |
| 2  | PU.1       | macrophage                                                | 40673490 |
| 3  | PAX5       | B cell                                                    | 39424985 |
| 4  | FOXP3      | Treg regulatory T cell                                    | 36569832 |
| 5  | MYOD1      | muscle cell muscle fiber myocyte                          | 36739949 |
| 6  | NEUROD1    | neuron nerve cell                                         | 39105169 |
| 7  | NKX2-1     | AT2 cell alveolar type II cell alveolar type 2 cell       | 39284798 |
| 8  | RUNX2      | osteoblast                                                | 39337587 |
| 9  | OLIG2      | oligodendrocyte oligodendroglia OPC oligodendro           | 34172044 |
| 10 | CEBPA      | neutrophil granulocyte                                    | 24584857 |
| 11 | NOTCH1     | leukemia AML ALL CML CLL                                  | 40009499 |
| 12 | PD-1       | melanoma                                                  | 30089911 |
| 13 | PD-L1      | non-small cell lung cancer NSCLC                          | 35159131 |
| 14 | KLF4       | colorectal cancer CRC                                     | 37031197 |
| 15 | FOXA1      | prostate cancer                                           | 32946061 |
| 16 | ESR1       | breast cancer                                             | 32289273 |
| 17 | APOE4      | Alzheimer's disease Alzheimer disease AD                  | 29861287 |
| 18 | TCF7L2     | diabetes mellitus diabetes T2D T2DM                       | 34016596 |
| 19 | FOXR2      | neuroblastoma NB-FOXR2                                    | 26919435 |
| 20 | PAX3-FOXO1 | rhabdomyosarcoma                                          | 37968277 |

## Pipeline

### Step 1: Run Inspect Modes

**Purpose:** Generate inspection reports for each method against the target paper.  
**Input:** Gene, context, and target PMID.  
**Output:** Method-specific output directories and log files.

**Command:**
```bash
GENE="FOXA1"
CONTEXTS=("prostate cancer")
PMID=32946061

{ time geneknow inspect -g $GENE -c "${CONTEXTS[@]}" -o . -n GeneKnow_inspect --max-passages 3 --pmid $PMID; } &> geneknow_inspect.log

{ time python ${ABLATION}/naive_inspect_1.py -g $GENE -c "${CONTEXTS[@]}" -o . -n naive_inspect_1 --pmid $PMID; } &> naive_inspect_1.log

{ time python ${ABLATION}/naive_inspect_2.py -g $GENE -c "${CONTEXTS[@]}" -o . -n naive_inspect_2 --max-passages 3 --pmid $PMID; } &> naive_inspect_2.log
```

---

### Step 2: Evaluate Alignment

**Purpose:** Assess how well each method's extracted claims align with the gene and context.  
**Input:** Method output directories.  
**Output:** `alignment_eval/`

**Command:**
```bash
methods=("GeneKnow_inspect" "naive_inspect_1" "naive_inspect_2")

for m in "${methods[@]}"; do
  python ${SCRIPTS}/inspect_alignment_eval.py --input_dir ${m} --output_dir alignment_eval --gene $GENE --constraint "${CONTEXTS[@]}" --model "gpt-5"
done
```

---

### Step 3: Generate and Refine Nugget Questions

**Purpose:** Create a curated set of yes/no questions that probe the gene's key roles and functions.  
**Input:** `alignment_eval/`  
**Output:** `candidate_nugget_questions.csv`, `nugget_questions.csv`

**Commands:**
```bash
python ${SCRIPTS}/generate_nugget_questions.py --input_dir alignment_eval --gene $GENE --max_questions 100

python ${SCRIPTS}/refine_nugget_questions.py --input candidate_nugget_questions.csv --gene $GENE --model "gpt-5"
```

**Note:** Manually curate the candidate questions: delete redundant items and keep only questions directly related to the gene's key roles and functions.

---

### Step 4: Evaluate Coverage

**Purpose:** Measure coverage of the nugget questions.  
**Input:** Method output directories and refined nugget questions.  
**Output:** `coverage_eval/`

**Command:**
```bash
for m in "${methods[@]}"; do
  python ${SCRIPTS}/inspect_coverage_eval.py --input_dir ${m} --output_dir coverage_eval --model "gpt-5"
done
```

### Step 5: Visualization

**Purpose:** Aggregate alignment, coverage, and F1 scores across all cases and generate summary plots.  
**Input:** `inspect_cases/case*/alignment_eval/` and `inspect_cases/case*/coverage_eval/`  
**Output:** PNG plots (box plots and stacked bar chart).

**Command:**
```bash
Rscript ${SCRIPTS}/../visualization/inspect_visualization.R
```

**What it plots:**
- Per-case alignment, coverage, and F1 scores (box plots).
- Claim type breakdown per method (stacked bar chart).

---

## Appendix: Single-Method Pipeline

If you prefer to evaluate one method at a time:

```bash
m="GeneKnow_inspect"

python ${SCRIPTS}/inspect_alignment_eval.py --input_dir ${m} --output_dir alignment_eval --gene $GENE --constraint "${CONTEXTS[@]}" --model "gpt-5"

# If nugget questions are regenerated, all subsequent steps for every method must be re-run.
python ${SCRIPTS}/generate_nugget_questions.py --input_dir alignment_eval

python ${SCRIPTS}/refine_nugget_questions.py --input candidate_nugget_questions.csv --gene $GENE --model "gpt-5"

python ${SCRIPTS}/inspect_coverage_eval.py --input_dir ${m} --output_dir coverage_eval --model "gpt-5"
```
