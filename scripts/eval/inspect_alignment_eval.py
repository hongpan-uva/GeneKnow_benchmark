#!/usr/bin/env python3
"""
Evaluate alignment between a summary and evidence.

This script extracts claims from a summary, generates questions from those claims,
and checks if the evidence supports them (measures precision/alignment).

Usage:
    python inspect_alignment_eval.py --input_dir <path> --output_dir <path> \
                                     --gene <gene_name> --constraints <constraint1> [<constraint2> ...] \
                                     [--model <model>]

Example:
    python inspect_alignment_eval.py --input_dir ../inspect_cases/case1/naive_inspect_2 \
                                     --output_dir ./results \
                                     --gene SOX9 --constraints astrocyte
    
    python inspect_alignment_eval.py --input_dir ../inspect_cases/case1/naive_inspect_2 \
                                     --output_dir ./results \
                                     --gene SOX9 --constraints astrocyte neuron glial
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from openai import OpenAI
from jsonschema import validate

from generate_questions import generate_questions
from verify_claims import answer_questions_batch
from token_tracker import TokenTracker


def load_summary(summary_file: str) -> str:
    """
    Load summary text from GeneKnow_report.csv.

    Args:
        summary_file: Path to GeneKnow_report.csv

    Returns:
        Summary text string
    """
    df = pd.read_csv(summary_file)
    if 'Summary' not in df.columns:
        print(f"Error: 'Summary' column not found in {summary_file}")
        sys.exit(1)

    summary = df.iloc[0]['Summary']
    return str(summary)


def load_evidence(evidence_file: str) -> str:
    """
    Load and concatenate all evidence passages.

    Args:
        evidence_file: Path to evidence_passages CSV file

    Returns:
        Concatenated evidence text string
    """
    df = pd.read_csv(evidence_file)
    if 'text' not in df.columns:
        print(f"Error: 'text' column not found in {evidence_file}")
        sys.exit(1)

    # Concatenate all text passages with newlines
    evidence = "\n\n".join(df['text'].astype(str).tolist())
    return evidence


def generate_prompt_insert(g_primary_name, g_secondary_names, ct_all_names):
    if len(g_secondary_names) == 0:
        g_insert = g_primary_name
    else:
        g_insert = g_primary_name + " (" + "aliases: " + \
            ", ".join(g_secondary_names) + ")"

    if len(ct_all_names) == 1:
        ct_insert = ct_all_names[0]
    else:
        ct_insert = ct_all_names[0] + \
            " (" + "aliases: " + ", ".join(ct_all_names[1:]) + ")"

    return ({"g_insert": g_insert, "ct_insert": ct_insert})


def extract_claims(passage: str, tracker: TokenTracker, prompt_paramters: dict, model="gpt-5"):
    """
    Extract claims from a paragraph
    """
    client = prompt_paramters["client"]

    output_schema = {
        "title": "ClaimsObject",
        "type": "array",
        "items": {
            "type": "object",
            "required": ["id", "claim"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "integer"},
                "claim": {"type": "string"}
            }
        }
    }

    response = client.responses.create(
        model=model,
        input=[{"role": "system",
                "content": "You are a claim extractor. Extract atomic, checkable claims from scientific summary."},
               {"role": "developer",
               "content": f"""
A claim is defined as one proposition anchored to a single subject-predicate relationship.

Instructions:
- Extract atomic, checkable claims that exactly reflect the source—no external facts, inference, or speculation.
- Preserve hedging/negation/conditions, entity names/aliases as written, and any numbers, units, directions, or temporal cues.
- Replace all pronouns and demonstratives such as "this population", "these cells" with the full explicit entity name described in the source text.
- Split long or compound sentences into multiple claims.
- Each claim must be self-contained and interpretable in isolation, restate necessary constraints, context, and entity names explicitly
- Use the fewest non-overlapping claims needed to fully cover the text.
- Return the extracted claims with sequential ids in a JSON array, example:
[
        {{
            "id" : 1,
            "claim" : "claim1"
        }},
        {{
            "id" : 2,
            "claim" : "claim2"
        }},
        {{
            "id" : 3,
            "claim" : "claim3"
        }}
]

- Retrun the following blank JSON array, if no claims was extracted or the source text is blank:
[]
"""},
               {"role": "user",
                "content": [
                    {"type": "input_text", "text": f"Extract checkable claims from the following text"
                     },
                    {"type": "input_text", "text": passage}
                ]
                }]
    )

    tracker.update(response.usage)
    structured_output = json.loads(response.output_text)

    validate(instance=structured_output, schema=output_schema)

    return structured_output


def calculate_alignment(claim_answers: list, evidence_answers: list) -> float:
    """
    Calculate alignment score between claim answers and evidence answers.

    Alignment measures how often the evidence agrees with the claims.
    Only claims with non-"Idk" answers from the summary are considered valid.

    Formula: (# of matching answers) / (# of valid claims)
    Valid claims = those where claim_answer != "Idk"
    Matching = claim_answer == evidence_answer

    Args:
        claim_answers: List of answer strings from claims (can be "Yes", "No", or "Idk")
        evidence_answers: List of answer strings from evidence ("Yes", "No", or "Idk")

    Returns:
        Alignment score (0.0 to 1.0)

    Note:
        The claim_answers and evidence_answers lists must have the same length
        and be in matched order (i.e., claim_answers[i] corresponds to evidence_answers[i]).
    """
    valid_count = 0
    true_count = 0

    for ans1, ans2 in zip(claim_answers, evidence_answers):
        if ans1 != "Idk":
            valid_count += 1
            if ans1 == ans2:
                true_count += 1

    if valid_count == 0:
        return 0.0

    return true_count / valid_count


def discover_files(input_dir: Path) -> tuple:
    """
    Discover summary and evidence files in input directory.

    Args:
        input_dir: Path to input directory

    Returns:
        Tuple of (summary_file_path, evidence_file_path)

    Raises:
        SystemExit: If files not found or validation fails
    """
    # Check input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Find summary file
    summary_file = input_dir / "GeneKnow_report.csv"
    if not summary_file.exists():
        print(f"Error: Summary file not found: {summary_file}")
        sys.exit(1)

    # Find evidence directory and files
    evidence_dir = input_dir / "evidence_passages"
    if not evidence_dir.exists():
        print(f"Error: Evidence passages directory not found: {evidence_dir}")
        sys.exit(1)

    evidence_files = list(evidence_dir.glob("*_evidence_passages.csv"))

    if len(evidence_files) == 0:
        print(f"Error: No evidence passages file found in {evidence_dir}")
        print(f"Expected file pattern: *_evidence_passages.csv")
        sys.exit(1)

    if len(evidence_files) > 1:
        print(
            f"Error: Multiple evidence passages files found in {evidence_dir}")
        print(f"Found {len(evidence_files)} files:")
        for f in evidence_files:
            print(f"  - {f.name}")
        print(f"Expected exactly one file matching pattern: *_evidence_passages.csv")
        sys.exit(1)

    evidence_file = evidence_files[0]

    return str(summary_file), str(evidence_file)


def run_alignment_evaluation(input_dir: Path, output_dir: Path,
                             gene: str, constraints: list,
                             model: str = "gpt-5-mini") -> None:
    """
    Run the alignment evaluation workflow.

    Args:
        input_dir: Path to input directory containing GeneKnow_report.csv and evidence_passages/
        output_dir: Directory to save output files
        gene: Gene name for prompt generation
        constraints: List of constraints/contexts for prompt generation
        model: LLM model to use
    """
    # Discover files
    summary_file, evidence_file = discover_files(input_dir)

    # Initialize
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    tracker = TokenTracker()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Loading summary from: {summary_file}")
    summary = load_summary(summary_file)
    print(f"Summary length: {len(summary)} characters")

    print(f"\nLoading evidence from: {evidence_file}")
    evidence = load_evidence(evidence_file)
    print(f"Evidence length: {len(evidence)} characters")

    # Create prompt parameters
    inserts = generate_prompt_insert(gene, [], constraints)
    prompt_parameters = {
        "g_insert": inserts["g_insert"],
        "ct_insert": inserts["ct_insert"],
        "client": client
    }

    # Step 1: Extract claims from summary
    print(f"\nExtracting claims from summary using {model}...")
    claims = extract_claims(summary, tracker, prompt_parameters, model=model)
    print(f"Extracted {len(claims)} claims")

    # Determine output filename: <input_dir_name>_claims.csv
    output_filename = f"{input_dir.name}_claims.csv"
    claims_file = output_dir / output_filename

    if len(claims) == 0:
        print("Warning: No claims extracted. Alignment = 0")
        # Create empty results file
        columns = ['id', 'claim', 'question', 'claim_answer',
                   'evidence_answer', 'answer_note']
        empty_df = pd.DataFrame(columns=columns)  # type: ignore
        empty_df.to_csv(claims_file, index=False)
        print(f"\nEmpty results saved to: {claims_file}")
        tracker.print_report()
        return

    # Step 2: Save claims and generate questions
    pd.DataFrame(claims).to_csv(claims_file, index=False)
    print(f"\nGenerating questions from claims using {model}...")
    generate_questions(str(claims_file), model=model, tracker=tracker)

    # Step 3: Read back the questions
    claims_df = pd.read_csv(claims_file)
    questions = []
    for _, row in claims_df.iterrows():
        question_val = row.get('question')
        is_not_na = bool(pd.notna(question_val))
        if is_not_na and str(question_val).strip():
            questions.append({
                'id': int(row['id']),
                'question': str(row['question'])
            })

    print(f"Generated {len(questions)} valid questions")

    if len(questions) == 0:
        print("Warning: No valid questions generated. Alignment = 0")
        # Add empty columns
        claims_df['evidence_answer'] = ''
        claims_df['answer_note'] = ''
        claims_df.to_csv(claims_file, index=False)
        print(f"\nResults saved to: {claims_file}")
        tracker.print_report()
        return

    # Step 4: Answer questions based on evidence
    print(f"\nAnswering questions based on evidence using {model}...")
    evidence_answers = answer_questions_batch(
        questions, evidence, model=model, client=client, tracker=tracker
    )
    print(f"Received {len(evidence_answers)} answers from evidence")

    # Step 5: Merge results into final DataFrame
    # Create lookup dict for evidence answers
    evidence_lookup = {a['id']: a for a in evidence_answers}

    # Add evidence-based columns
    claims_df['evidence_answer'] = claims_df['id'].map(
        lambda x: evidence_lookup.get(x, {}).get('answer', '')
    )
    claims_df['answer_note'] = claims_df['id'].map(
        lambda x: evidence_lookup.get(x, {}).get('explanation', '')
    )

    # Ensure claim_answer column exists (should be added by generate_questions)
    if 'claim_answer' not in claims_df.columns:
        claims_df['claim_answer'] = 'Yes'

    # Reorder columns to match expected format
    columns_order = ['id', 'claim', 'question', 'claim_answer',
                     'evidence_answer', 'answer_note']
    claims_df = claims_df[[
        col for col in columns_order if col in claims_df.columns]]

    # Save final results
    claims_df.to_csv(claims_file, index=False)
    print(f"\nResults saved to: {claims_file}")

    # Step 6: Calculate and print alignment
    alignment = calculate_alignment(
        claims_df['claim_answer'].tolist(),
        claims_df['evidence_answer'].tolist())
    yes_count = sum(1 for a in evidence_answers if a.get('answer') == 'Yes')

    print(f"\n{'='*60}")
    print(f"ALIGNMENT RESULTS")
    print(f"{'='*60}")
    print(f"Total claims: {len(claims)}")
    print(f"Claims supported by evidence: {yes_count}")
    print(f"Alignment (Precision): {alignment:.2%}")
    print(f"{'='*60}")

    # Print token usage
    tracker.print_report()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate alignment between a summary and evidence."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to input directory containing GeneKnow_report.csv and evidence_passages/"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--gene",
        required=True,
        help="Gene name (e.g., SOX9)"
    )
    parser.add_argument(
        "--constraints",
        nargs='+',
        required=True,
        help="One or more constraints/contexts (e.g., astrocyte neuron glial)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="LLM model to use (default: gpt-5-mini)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    run_alignment_evaluation(
        input_dir=input_dir,
        output_dir=output_dir,
        gene=args.gene,
        constraints=args.constraints,
        model=args.model
    )


if __name__ == "__main__":
    main()
