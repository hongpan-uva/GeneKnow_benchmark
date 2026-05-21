#!/usr/bin/env python3
"""
Generate nugget questions from claim files.

This script reads claim files from a folder, filters for valid rows (where both
claim_answer and evidence_answer are "Yes"), optionally filters by gene and/or
constraints, selects questions using round-robin across files (avoiding redundant
claims), reorders IDs sequentially from 1, and creates a consolidated nugget
questions file.

Usage:
    python generate_nugget_questions.py --input_dir <path> [--gene <name>] [--constraints <terms>] [--max_questions <n>]

Examples:
    # Gene only
    python generate_nugget_questions.py --input_dir ../inspect_cases/case1/summary_claims/ --gene SOX9
    
    # Constraints only
    python generate_nugget_questions.py --input_dir ../inspect_cases/case1/summary_claims/ --constraints astrocyte neuron
    
    # Both gene and constraints, with custom max
    python generate_nugget_questions.py --input_dir ../inspect_cases/case1/summary_claims/ --gene SOX9 --constraints astrocyte --max_questions 15
    
    # No filtering
    python generate_nugget_questions.py --input_dir ../inspect_cases/case1/summary_claims/

Output:
    candidate_nugget_questions.csv (saved in current directory)
"""

import argparse
import re
import sys
from pathlib import Path
import pandas as pd


def word_matches_prefix(text: str, prefix: str) -> bool:
    """
    Check if text contains word starting with prefix (case-insensitive).

    Args:
        text: Text to search in
        prefix: Prefix to match

    Returns:
        True if any word in text starts with prefix
    """
    text_lower = text.lower()
    prefix_lower = prefix.lower()
    # Use regex to find if prefix appears at word boundary
    pattern = r'\b' + re.escape(prefix_lower) + r'\w*'
    return re.search(pattern, text_lower) is not None


def question_matches_filter(question: str, gene: str, constraints: list) -> bool:
    """
    Check if question matches the filter criteria.

    Matches if:
    - Gene is provided and matches, OR
    - Any constraint matches

    Args:
        question: Question text to check
        gene: Gene name (optional)
        constraints: List of constraint terms (optional)

    Returns:
        True if question matches filter
    """
    # If no filter criteria provided, don't filter
    if not gene and not constraints:
        return True

    # Check gene match
    if gene and word_matches_prefix(question, gene):
        return True

    # Check constraint matches
    if constraints and any(word_matches_prefix(question, c) for c in constraints):
        return True

    # Filter criteria provided but no match
    return False


def is_redundant(claim: str, selected_claims: set) -> bool:
    """
    Check if claim is redundant (exact match, case-insensitive) with already selected claims.

    Args:
        claim: Claim text to check
        selected_claims: Set of already selected claim texts (normalized to lowercase)

    Returns:
        True if claim is redundant
    """
    return claim.lower() in selected_claims


def discover_claim_files(input_dir: Path) -> list:
    """
    Discover all *_claims.csv files in input directory.

    Args:
        input_dir: Path to input directory

    Returns:
        List of Path objects for claim files

    Raises:
        SystemExit: If no files found or directory doesn't exist
    """
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    claim_files = list(input_dir.glob("*_claims.csv"))

    if len(claim_files) == 0:
        print(f"Error: No *_claims.csv files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(claim_files)} claim file(s):")
    for f in sorted(claim_files):
        print(f"  - {f.name}")

    return sorted(claim_files)


def load_and_filter_file(file_path: Path, gene: str, constraints: list) -> pd.DataFrame:
    """
    Load a claim file and filter for valid rows that pass gene/constraint filters.

    Args:
        file_path: Path to claim CSV file
        gene: Gene name for filtering (optional)
        constraints: List of constraint terms for filtering (optional)

    Returns:
        DataFrame with valid, filtered rows

    Raises:
        SystemExit: If required columns missing
    """
    # Read file
    df = pd.read_csv(file_path)

    # Check required columns
    required_cols = ['id', 'claim', 'question', 'claim_answer',
                     'evidence_answer', 'answer_note']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Error: Missing required columns in {file_path.name}: {missing_cols}")
        sys.exit(1)

    # Filter valid rows: both claim_answer and evidence_answer are "Yes"
    valid_rows = df[(df['claim_answer'] == 'Yes') &
                    (df['evidence_answer'] == 'Yes')].copy()

    # Apply gene/constraint filtering if specified
    if gene or constraints:
        valid_rows = valid_rows[valid_rows['question'].apply(
            lambda q: question_matches_filter(q, gene, constraints)
        )].copy()

    return valid_rows


def select_questions_round_robin(files_data: list, file_names: list,
                                 max_questions: int) -> list:
    """
    Select questions using round-robin approach across files.

    In each round, pick the first unpicked, non-redundant question from each file,
    then move to the next file. Continue until max_questions reached or all files exhausted.

    Args:
        files_data: List of DataFrames, one per file
        file_names: List of file names (for reporting)
        max_questions: Maximum number of questions to select

    Returns:
        List of selected rows (as Series)
    """
    selected = []
    selected_claims = set()  # Normalized (lowercase) claim texts

    # Track which row indices have been used from each file
    file_row_indices = [set() for _ in range(len(files_data))]
    files_exhausted = [False] * len(files_data)

    current_file_idx = 0

    print(f"\nSelecting up to {max_questions} questions using round-robin...")

    while len(selected) < max_questions and not all(files_exhausted):
        file_idx = current_file_idx % len(files_data)

        # Skip exhausted files
        if files_exhausted[file_idx]:
            current_file_idx += 1
            continue

        df = files_data[file_idx]
        found_one = False

        # Find first unpicked, non-redundant question in this file
        for row_idx, row in df.iterrows():
            if row_idx in file_row_indices[file_idx]:
                continue

            # Mark as checked
            file_row_indices[file_idx].add(row_idx)

            # Check if redundant
            if is_redundant(row['claim'], selected_claims):
                continue

            # Found one! Add to selection
            selected.append(row)
            selected_claims.add(row['claim'].lower())
            found_one = True
            print(
                f"  [{len(selected)}/{max_questions}] From {file_names[file_idx]}: {row['question'][:60]}...")
            break

        if not found_one:
            # This file has no more valid questions
            files_exhausted[file_idx] = True
            print(f"  File {file_names[file_idx]} exhausted")

        current_file_idx += 1

    return selected


def generate_nugget_questions(input_dir: Path, output_file: Path,
                              gene: str, constraints: list, max_questions: int) -> None:
    """
    Main function to generate nugget questions from claim files.

    Args:
        input_dir: Directory containing *_claims.csv files
        output_file: Path to output CSV file
        gene: Gene name for filtering (optional)
        constraints: List of constraint terms for filtering (optional)
        max_questions: Maximum number of questions to select
    """
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Max questions: {max_questions}")

    if gene:
        print(f"Gene filter: {gene}")
    if constraints:
        print(f"Constraint filters: {constraints}")
    if not gene and not constraints:
        print("No filtering applied")

    # Discover claim files
    claim_files = discover_claim_files(input_dir)

    # Load and filter all files
    print("\nLoading and filtering files...")
    files_data = []
    file_names = []
    total_valid = 0

    for file_path in claim_files:
        df = load_and_filter_file(file_path, gene, constraints)
        if len(df) > 0:
            files_data.append(df)
            file_names.append(file_path.name)
            total_valid += len(df)
            print(f"  {file_path.name}: {len(df)} valid rows")

    if len(files_data) == 0:
        print("\nError: No valid questions found in any file")
        sys.exit(1)

    print(f"\nTotal valid questions across all files: {total_valid}")

    # Select questions using round-robin
    selected_rows = select_questions_round_robin(
        files_data, file_names, max_questions)

    if len(selected_rows) == 0:
        print("\nError: No non-redundant questions could be selected")
        sys.exit(1)

    # Create DataFrame from selected rows
    combined_df = pd.DataFrame(selected_rows)

    # Reorder IDs sequentially from 1
    combined_df['id'] = range(1, len(combined_df) + 1)

    # Clear evidence_answer and answer_note, ensure claim_answer is "Yes"
    combined_df['evidence_answer'] = ""
    combined_df['answer_note'] = ""
    combined_df['claim_answer'] = "Yes"

    # Ensure column order
    columns_order = ['id', 'claim', 'question', 'claim_answer',
                     'evidence_answer', 'answer_note']
    combined_df = combined_df[columns_order]

    # Save to CSV
    combined_df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"SUCCESS")
    print(f"{'='*60}")
    print(f"Files processed: {len(claim_files)}")
    print(f"Files with valid questions: {len(files_data)}")
    print(f"Total nugget questions selected: {len(combined_df)}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate nugget questions from claim files using round-robin selection."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to directory containing *_claims.csv files"
    )
    parser.add_argument(
        "--gene",
        default=None,
        help="Gene name to filter by (optional)"
    )
    parser.add_argument(
        "--constraints",
        nargs='+',
        default=None,
        help="One or more constraint terms to filter by (optional)"
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=20,
        help="Maximum number of questions to select (default: 10)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path("candidate_nugget_questions.csv")

    generate_nugget_questions(input_dir, output_file,
                              args.gene, args.constraints, args.max_questions)


if __name__ == "__main__":
    main()
