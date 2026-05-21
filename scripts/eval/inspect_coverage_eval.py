#!/usr/bin/env python3
"""
Evaluate coverage (recall) of a summary against nugget questions.

This script reads nugget questions and a summary, answers the questions based on
the summary, and calculates coverage (recall) by comparing summary answers with
expected answers.

Usage:
    python inspect_coverage_eval.py --input_dir <path> --output_dir <path> [--model <model>]

Example:
    python inspect_coverage_eval.py --input_dir ../inspect_cases/case1 --output_dir ./results

Output CSV columns:
    - id: Question ID
    - claim: Claim text
    - question: Generated question
    - claim_answer: Expected answer (from nugget file)
    - summary_answer: Answer based on summary (Yes/No/Idk)
    - answer_note: Explanation for summary answer
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from openai import OpenAI

from inspect_alignment_eval import load_summary
from verify_claims import answer_questions_batch
from token_tracker import TokenTracker


def discover_files(input_dir: Path) -> tuple:
    """
    Discover nugget_questions.csv and GeneKnow_report.csv in input directory.

    Args:
        input_dir: Path to input directory

    Returns:
        Tuple of (nugget_file_path, summary_file_path)

    Raises:
        SystemExit: If files not found
    """
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Find summary file
    summary_file = input_dir / "GeneKnow_report.csv"
    if not summary_file.exists():
        print(f"Error: Summary file not found: {summary_file}")
        sys.exit(1)

    return str(summary_file)


def calculate_coverage(claim_answers: list, summary_answers: list) -> float:
    """
    Calculate coverage (recall) score.

    Coverage = (# of matched answers) / (# of valid questions)
    Valid questions = those where claim_answer != "Idk"
    Matched = claim_answer == summary_answer

    Args:
        claim_answers: List of expected answer strings from nugget file
        summary_answers: List of answer strings from summary

    Returns:
        Coverage score (0.0 to 1.0)

    Note:
        The claim_answers and summary_answers lists must have the same length
        and be in matched order.
    """
    valid_count = 0
    matched_count = 0

    for claim_ans, summary_ans in zip(claim_answers, summary_answers):
        if claim_ans != "Idk":
            valid_count += 1
            if claim_ans == summary_ans:
                matched_count += 1

    if valid_count == 0:
        return 0.0

    return matched_count / valid_count


def run_coverage_evaluation(input_dir: Path, output_dir: Path,
                            model: str = "gpt-5-mini") -> None:
    """
    Run the coverage evaluation workflow.

    Args:
        input_dir: Path to input directory containing nugget_questions.csv and GeneKnow_report.csv
        output_dir: Directory to save output files
        model: LLM model to use
    """
    # Find nugget questions file
    nugget_file = Path("nugget_questions.csv")
    if not nugget_file.exists():
        print(f"Error: Nugget questions file not found: {nugget_file}")
        sys.exit(1)

    # Discover files
    summary_file = discover_files(input_dir)

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
    print(f"Loading nugget questions from: {nugget_file}")

    # Load nugget questions
    nugget_df = pd.read_csv(nugget_file)

    # Validate required columns
    required_cols = ['id', 'claim', 'question', 'claim_answer']
    missing_cols = [
        col for col in required_cols if col not in nugget_df.columns]
    if missing_cols:
        print(
            f"Error: Missing required columns in nugget file: {missing_cols}")
        sys.exit(1)

    print(f"Loaded {len(nugget_df)} nugget questions")

    # Load summary
    print(f"\nLoading summary from: {summary_file}")
    summary = load_summary(summary_file)
    print(f"Summary length: {len(summary)} characters")

    # Prepare questions
    questions = []
    for _, row in nugget_df.iterrows():
        question_val = row.get('question')
        is_not_na = bool(pd.notna(question_val))
        if is_not_na and str(question_val).strip():
            questions.append({
                'id': int(row['id']),
                'question': str(row['question'])
            })

    print(f"\nPrepared {len(questions)} valid questions")

    if len(questions) == 0:
        print("Warning: No valid questions found. Coverage = 0")
        # Create empty output file
        output_filename = f"{input_dir.name}_coverage.csv"
        output_file = output_dir / output_filename
        empty_df = pd.DataFrame(columns=['id', 'claim', 'question', 'claim_answer',
                                         'summary_answer', 'answer_note'])
        empty_df.to_csv(output_file, index=False)
        print(f"\nEmpty results saved to: {output_file}")
        tracker.print_report()
        return

    # Step 1: Answer questions based on summary
    print(f"\nAnswering questions based on summary using {model}...")
    summary_answers = answer_questions_batch(
        questions, summary, model=model, client=client, tracker=tracker
    )
    print(f"Received {len(summary_answers)} answers from summary")

    # Step 2: Merge results into DataFrame
    # Create lookup dict for summary answers
    summary_lookup = {a['id']: a for a in summary_answers}

    # Add summary-based columns
    nugget_df['summary_answer'] = nugget_df['id'].map(
        lambda x: summary_lookup.get(x, {}).get('answer', '')
    )
    nugget_df['answer_note'] = nugget_df['id'].map(
        lambda x: summary_lookup.get(x, {}).get('explanation', '')
    )

    # Reorder columns to match expected format
    columns_order = ['id', 'claim', 'question', 'claim_answer',
                     'summary_answer', 'answer_note']
    nugget_df = nugget_df[[
        col for col in columns_order if col in nugget_df.columns]]

    # Save results
    output_filename = f"{input_dir.name}_coverage.csv"
    output_file = output_dir / output_filename
    nugget_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Step 3: Calculate and print coverage
    coverage = calculate_coverage(
        nugget_df['claim_answer'].tolist(),
        nugget_df['summary_answer'].tolist())
    matched_count = sum(
        1 for claim_ans, summary_ans in zip(nugget_df['claim_answer'], nugget_df['summary_answer'])
        if claim_ans == summary_ans and claim_ans != "Idk")

    print(f"\n{'='*60}")
    print(f"COVERAGE RESULTS")
    print(f"{'='*60}")
    print(f"Total questions: {len(nugget_df)}")
    print(f"Matched answers: {matched_count}")
    print(f"Coverage (Recall): {coverage:.2%}")
    print(f"{'='*60}")

    # Print token usage
    tracker.print_report()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate coverage (recall) of a summary against nugget questions."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to input directory containing nugget_questions.csv and GeneKnow_report.csv"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save output files"
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="LLM model to use (default: gpt-5-mini)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    run_coverage_evaluation(
        input_dir=input_dir,
        output_dir=output_dir,
        model=args.model
    )


if __name__ == "__main__":
    main()
