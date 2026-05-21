# Discover Benchmark Workflow

Open-domain retrieval and synopsis generation benchmark.
This workflow evaluates how well a method retrieves literature and generates a concise, cited synopsis describing the functional relationship between a gene and a cell type or disease context.

## Prerequisites

- Python 3.x
- `geneknow` CLI installed and on `PATH`
- `OPENAI_API_KEY` environment variable set
- Set a working variable for the evaluation script directory:
  ```bash
  SCRIPTS=/path/to/evaluation/scripts/eval
  ```

## Test Cases

| id   | gene       | alias  | context                                             |
| ---- | ---------- | ------ | --------------------------------------------------- |
| 1    | SOX9       | —      | astrocyte astrocytic                                |
| 2    | PU.1       | SPI1   | macrophage                                          |
| 3    | PAX5       | —      | B cell                                              |
| 4    | FOXP3      | —      | Treg regulatory T cell                              |
| 5    | MYOD1      | —      | muscle cell muscle fiber myocyte                    |
| 6    | NEUROD1    | NEUROD | neuron nerve cell                                   |
| 7    | NKX2-1     | TTF-1  | AT2 cell alveolar type II cell alveolar type 2 cell |
| 8    | RUNX2      | —      | osteoblast                                          |
| 9    | OLIG2      | —      | oligodendrocyte oligodendroglia OPC oligodendro     |
| 10   | CEBPA      | —      | neutrophil granulocyte                              |
| 11   | NOTCH1     | —      | leukemia AML ALL CML CLL                            |
| 12   | PD-1       | —      | melanoma                                            |
| 13   | PD-L1      | —      | non-small cell lung cancer NSCLC                    |
| 14   | KLF4       | —      | colorectal cancer CRC                               |
| 15   | FOXA1      | —      | prostate cancer                                     |
| 16   | ESR1       | —      | breast cancer                                       |
| 17   | APOE4      | —      | Alzheimer's disease Alzheimer disease AD            |
| 18   | TCF7L2     | —      | diabetes mellitus diabetes T2D T2DM                 |
| 19   | FOXR2      | —      | neuroblastoma NB-FOXR2                              |
| 20   | PAX3-FOXO1 | FKHR   | rhabdomyosarcoma                                    |

Aliases are used only in verification scripts.

## Pipeline

### Step 1: Run Methods

**Purpose:** Generate synopses for each method being evaluated.  
**Input:** Gene, context, and method-specific parameters.  
**Output:** Text files in `outs/`.

**Commands:**

GeneKnow (substitute `<gene>`, `<context>`, and `<name>` as needed):
```bash
time geneknow discover -g <gene> -c <context> -o . -n <name> --max-passages 3 --max-papers 5
```

Copy the resulting synopsis into the shared output folder:
```bash
cp GeneKnow_out/synopses/<gene>_synopsis.txt outs/geneknow.txt
```

Prompt for chat tools (the exact prompt is also available in `discover_benchmark_chat_prompt.txt` at the repository root):
```text
Write a concise single-paragraph synopsis (~200 words) describing the functional relationship between {gene} and {cell_type}, including only claims directly supported by peer-reviewed research articles; do not infer, speculate, or generalize beyond what is explicitly reported. For each claim, cite a real publication that directly supports that specific statement, and present the references after the paragraph in APA 7th format with DOI links. If no direct evidence exists, explicitly state that instead of extrapolating.
```

Save all method outputs to `outs/` using consistent filenames (e.g., `outs/geneknow.txt`, `outs/chatgpt5.2_thinking.txt`).

---

### Step 2: Extract Citations and Claims

**Purpose:** Parse in-text citations and factual claims from each synopsis.  
**Input:** `outs/<method>.txt`  
**Output:** `references/<method>_references.csv`, `claims/<method>_claims.csv`

**Command:**
```bash
methods=("geneknow" "chatgpt5.2_thinking" "gemini3_thinking" "claude_opus4.6")

for m in "${methods[@]}"; do
  python ${SCRIPTS}/extract_citations.py --input outs/${m}.txt --output_dir references --model "gpt-5"
done

for m in "${methods[@]}"; do
  python ${SCRIPTS}/extract_claims.py --input outs/${m}.txt --output_dir claims --model "gpt-5"
done
```

---

### Step 3: Validate Citations Manually

**Purpose:** Confirm that extracted references are real and that DOI links are correct.  
**Input:** `references/<method>_references.csv`  
**Output:** Updated CSVs with `real_reference`, `real_url`, and `alternative_url` fields.

**Instructions:**
1. Open each generated CSV.
2. Search the reference title in PubMed to confirm the paper exists and the first author is correct; mark `true` or `false` in the `real_reference` field.
3. Click the DOI link to confirm it resolves to the correct paper; mark `true` or `false` in the `real_url` field.
4. If the paper is real but the DOI link is fake or paywalled, add an accessible URL in `alternative_url`.

---

### Step 4: Check Citation-Claim Consistency

**Purpose:** Ensure that every claim cites a reference present in the reference list.  
**Input:** `claims/`, `references/`  
**Output:** Console report.

**Command:**
```bash
python ${SCRIPTS}/check_citations.py --claim_path claims/ --ref_path references/
```

---

### Step 5: Download and Process Papers

**Purpose:** Prepare full-text source material for automated claim verification.  
**Input:** Validated references.  
**Output:** `references/papers/` (HTML and extracted Markdown).

**Commands:**
```bash
# Generate paper hash IDs
python ${SCRIPTS}/hash_url.py references

# Convert any PDFs to Markdown
python ${SCRIPTS}/convert_pdf_to_md.py references/papers

# Extract content from downloaded HTML/PDF files
python ${SCRIPTS}/batch_extract_content.py --input references/papers
```

**Note:** Manually download the HTML (or PDF) versions of each validated paper and save them to `references/papers/` before running `batch_extract_content.py`.

---

### Step 6: Verify Claims

**Purpose:** Generate yes/no questions from claims and verify them against the downloaded papers.  
**Input:** `claims/<method>_claims.csv`, `references/papers/`  
**Output:** Updated claims CSV with verification results.

**Commands:**
```bash
for m in "${methods[@]}"; do
  python ${SCRIPTS}/generate_questions.py claims/${m}_claims.csv --model "gpt-5"
done

for m in "${methods[@]}"; do
  python ${SCRIPTS}/verify_claims.py claims/${m}_claims.csv references references/papers --model "gpt-5"
done
```

### Step 7: Visualization

**Purpose:** Aggregate claim verification and reference validation results across all cases and generate summary plots.  
**Input:** `discover_cases/case*/claims/` and `discover_cases/case*/references/`  
**Output:** PNG plots (stacked bar charts and pie charts).

**Command:**
```bash
Rscript ${SCRIPTS}/../visualization/discover_visualization.R
```

**What it plots:**
- Claim verification outcomes per method (supported, opposed, unsure, non-verifiable).
- Reference validation outcomes per method (real paper & URL, fake paper, fake URL).

---

## Appendix: Single-Method Pipeline

If you prefer to run one method at a time:

```bash
m="geneknow"

python ${SCRIPTS}/extract_citations.py --input outs/${m}.txt --output_dir references --model "gpt-5"
python ${SCRIPTS}/extract_claims.py --input outs/${m}.txt --output_dir claims --model "gpt-5"

# Perform manual citation validation here

python ${SCRIPTS}/check_citations.py --claim_path claims/ --ref_path references/
python ${SCRIPTS}/hash_url.py references
python ${SCRIPTS}/convert_pdf_to_md.py references/papers
python ${SCRIPTS}/batch_extract_content.py --input references/papers

# Manually download papers into references/papers/

python ${SCRIPTS}/generate_questions.py claims/${m}_claims.csv --model "gpt-5"
python ${SCRIPTS}/verify_claims.py claims/${m}_claims.csv references references/papers --model "gpt-5"
```
