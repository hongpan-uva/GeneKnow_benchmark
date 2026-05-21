#!/usr/bin/env python3
"""
Refine nugget questions by extracting key claims about a gene's core role,
direct function, and impact, merging semantically equivalent claims within
each category, and regenerating questions.

This script reads a candidate nugget questions file, uses an LLM to:
1. Extract key claims (core role, direct function, impact) and discard
   overly specific mechanistic details.
2. Merge semantically equivalent claims within each category.
3. Generate YES/NO questions for the merged claims.

Usage:
    python refine_nugget_questions.py --input <path> --gene <name> [--model <model_name>]

Example:
    python refine_nugget_questions.py --input candidate_nugget_questions.csv --gene Sox9
    python refine_nugget_questions.py --input candidate_nugget_questions.csv --gene Sox9 --model gpt-5

Output:
    nugget_questions.csv (saved in current directory)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from jsonschema import validate
from openai import OpenAI

from generate_questions import generate_questions_batch
from token_tracker import TokenTracker


CATEGORY_ORDER = ["core_role", "direct_function", "impact", "other"]


def extract_and_merge_key_claims_with_llm(claims: list, ids: list, gene: str,
                                          model: str, client,
                                          tracker: TokenTracker) -> list:
    """
    Pass 1: Extract key claims (core role, direct function, impact),
    assign non-key claims to "other", and merge semantically equivalent claims
    within each category.

    Args:
        claims: List of claim strings
        ids: List of original IDs (same order as claims)
        gene: Gene name that the claims are about
        model: LLM model name
        client: OpenAI client
        tracker: TokenTracker instance

    Returns:
        List of dicts with keys 'category', 'source_ids' (list of int),
        and 'merged_claim' (str)
    """
    claims_json = [{"id": id_val, "claim": claim}
                   for id_val, claim in zip(ids, claims)]

    output_schema = {
        "title": "KeyClaimsArray",
        "type": "array",
        "items": {
            "type": "object",
            "required": ["category", "source_ids", "merged_claim"],
            "additionalProperties": False,
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["core_role", "direct_function", "impact", "other"]
                },
                "source_ids": {
                    "type": "array",
                    "items": {"type": "integer"}
                },
                "merged_claim": {"type": "string"}
            }
        }
    }

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a scientific curator specialized in gene function summarization."
            },
            {
                "role": "developer",
                "content": f"""
Task: From the provided candidate claims, extract key claims that describe {gene}'s:
1. Core role (fundamental identity/specification role, e.g., transcription factor, master regulator, cell type specifier)
2. Direct function (specific molecular or cellular activities the gene performs)
3. Impact (biological consequences, phenotypic outcomes, or disease relevance)

Claims that are overly specific mechanistic details not central to the gene's fundamental role (e.g., specific phosphorylation sites, intermediate molecular steps, subcellular localization details), expression pattern descriptions not directly tied to functional identity, redundant with higher-level claims already captured, or peripheral/tangential to the gene's main function should be assigned to category "other".

For each claim, assign it to one category: "core_role", "direct_function", "impact", or "other".
Then, within each category, group semantically equivalent claims (same underlying biological fact) and merge each group into ONE representative claim.

Merging rules (conservative):
- If claims mention different specific organisms, the merged claim should mention both specifically. Do NOT generalize to broader taxa unless the original claims already use those terms.
- If one claim is more specific than another, preserve the narrower scope.
- Combine overlapping facts into a single, concise sentence without losing specificity.
- Only merge claims that assert the same core fact within the same category.

Input format: JSON array of {{"id": int, "claim": str}}

Output format: JSON array of:
{{
  "category": "core_role" | "direct_function" | "impact" | "other",
  "source_ids": [list of original IDs in this group],
  "merged_claim": str
}}

Rules:
- Every input claim ID must appear in exactly one group's source_ids.
- A group can have one member (no merge needed).
- Return [] only if the input is empty.
"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract, categorize, and merge key claims from the following candidates:"
                    },
                    {
                        "type": "input_text",
                        "text": json.dumps(claims_json, indent=2)
                    }
                ]
            }
        ]
    )

    tracker.update(response.usage)

    structured_output = json.loads(response.output_text)
    validate(instance=structured_output, schema=output_schema)

    # Validate ID coverage
    all_source_ids = []
    for group in structured_output:
        all_source_ids.extend(group["source_ids"])

    expected_ids = set(ids)
    actual_ids = set(all_source_ids)

    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        msgs = []
        if missing:
            msgs.append(f"Missing IDs: {sorted(missing)}")
        if extra:
            msgs.append(f"Extra IDs: {sorted(extra)}")
        raise Exception("ID mismatch: " + "; ".join(msgs))

    if len(all_source_ids) != len(set(all_source_ids)):
        raise Exception("Duplicate IDs found in output groups")

    return structured_output


def sort_by_category(results: list) -> list:
    """
    Sort results by category order: core_role, direct_function, impact, other.
    """
    order_map = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
    return sorted(results, key=lambda x: order_map.get(x["category"], 999))


def main():
    parser = argparse.ArgumentParser(
        description="Refine nugget questions by extracting key claims about a "
                    "gene's core role, direct function, impact, and other, merging "
                    "semantically equivalent claims, and regenerating questions."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to candidate nugget questions CSV file"
    )
    parser.add_argument(
        "--gene",
        required=True,
        help="Gene name that the claims are about (e.g., Sox9)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="LLM model to use (default: gpt-5)"
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path("nugget_questions.csv")
    model = args.model
    gene = args.gene

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    print(f"Loading candidates from: {input_file}")
    df = pd.read_csv(input_file)

    if len(df) == 0:
        print("Error: Input file is empty")
        sys.exit(1)

    required_cols = ["id", "claim", "question", "claim_answer",
                     "evidence_answer", "answer_note"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        sys.exit(1)

    print(f"Found {len(df)} candidate questions")

    # Lightweight validation: warn if gene doesn't appear in claims
    gene_lower = gene.lower()
    gene_in_claims = sum(
        1 for c in df["claim"].astype(str) if gene_lower in c.lower()
    )
    if gene_in_claims == 0:
        print(
            f"\nWarning: Gene '{gene}' was not found in any claim text. "
            f"Please verify the gene name is correct."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    tracker = TokenTracker()

    # Pass 1: Extract and merge key claims
    print(f"\nPass 1: Extracting and merging key claims with {model}...")
    print(f"  Gene: {gene}")
    claims = df["claim"].astype(str).tolist()
    try:
        ids = df["id"].astype(int).tolist()
    except ValueError as e:
        print(f"Error: 'id' column must contain integers: {e}")
        sys.exit(1)

    try:
        merged_results = extract_and_merge_key_claims_with_llm(
            claims, ids, gene, model, client, tracker)
    except Exception as e:
        print(f"Error during claim extraction/merging: {e}")
        sys.exit(1)

    if len(merged_results) == 0:
        print("Error: No key claims were extracted")
        sys.exit(1)

    # Sort by category
    merged_results = sort_by_category(merged_results)

    print(f"  Extracted {len(merged_results)} key claim groups")
    for i, group in enumerate(merged_results, 1):
        source_str = ";".join(str(s) for s in group["source_ids"])
        print(f"  Group {i} ({group['category']}): IDs [{source_str}] -> "
              f"{group['merged_claim'][:60]}...")

    # Pass 2: Generate questions for merged claims
    print(f"\nPass 2: Generating questions for merged claims...")
    merged_claims = [group["merged_claim"] for group in merged_results]

    try:
        questions = generate_questions_batch(
            merged_claims, model, client, tracker)
        print("  Done")
    except Exception as e:
        print(f"Error during question generation: {e}")
        sys.exit(1)

    # Build output DataFrame
    print("\nBuilding refined output...")
    output_data = []
    for i, (group, question) in enumerate(zip(merged_results, questions), start=1):
        source_ids_str = ";".join(str(s) for s in group["source_ids"])
        output_data.append({
            "id": i,
            "claim": group["merged_claim"],
            "question": question,
            "claim_answer": "Yes",
            "evidence_answer": "",
            "answer_note": "",
            "category": group["category"],
            "source_ids": source_ids_str
        })

    output_df = pd.DataFrame(output_data)
    columns_order = [
        "id", "claim", "question", "claim_answer",
        "evidence_answer", "answer_note", "category", "source_ids"
    ]
    output_df = output_df[columns_order]
    output_df.to_csv(output_file, index=False)

    # Warn about empty questions
    empty_mask = output_df["question"].isna() | (
        output_df["question"].astype(str).str.strip() == "")
    if empty_mask.any():
        empty_ids = output_df.loc[empty_mask, "id"].tolist()
        print(
            f"\nWarning: {len(empty_ids)} question(s) are empty: IDs {empty_ids}")

    print(f"\n{'='*60}")
    print(f"SUCCESS")
    print(f"{'='*60}")
    print(f"Candidates: {len(df)}")
    print(f"Key claim groups: {len(merged_results)}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")

    tracker.print_report()


if __name__ == "__main__":
    main()
