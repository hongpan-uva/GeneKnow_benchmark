#!/usr/bin/env python3
"""
Verify claims against paper content using LLM.

Usage:
    python verify_claims.py <claims_file> <references_path> <papers_path> [--model <model_name>]

Example:
    python verify_claims.py ../discover_cases/case1/claims/geneknow_claims.csv \
                           ../discover_cases/case1/references \
                           ../discover_cases/case1/references/papers
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from openai import OpenAI
from jsonschema import validate
from html_processing import extract_full_text_from_html_file
from markdown_processing import extract_full_text_from_markdown_file
from token_tracker import TokenTracker


def extract_full_text_from_paper(hash_id: str, papers_path: str) -> str:
    """
    Extract full text from a paper file, trying multiple formats in order:
    1. .md (markdown - best extraction)
    2. .html (HTML)
    3. .htm (HTML)

    Args:
        hash_id: The hash ID of the paper
        papers_path: Path to papers directory

    Returns:
        Extracted text string

    Raises:
        FileNotFoundError: If no readable format found
        ValueError: If extraction fails
    """
    # Try markdown first (best extraction quality)
    md_file = Path(papers_path) / f"{hash_id}.md"
    if md_file.exists():
        text, metadata = extract_full_text_from_markdown_file(
            str(md_file), verbose=False)
        if metadata['success']:
            return text
        else:
            raise ValueError(
                f"Failed to extract text from markdown: {metadata.get('error', 'Unknown error')}")

    # Try HTML files
    for ext in ['.html', '.htm']:
        html_file = Path(papers_path) / f"{hash_id}{ext}"
        if html_file.exists():
            text, metadata = extract_full_text_from_html_file(
                str(html_file), verbose=False)
            if metadata['success']:
                return text
            else:
                raise ValueError(
                    f"Failed to extract text from HTML: {metadata.get('error', 'Unknown error')}")

    raise FileNotFoundError(
        f"No readable paper file found for {hash_id} (tried .md, .html, .htm)")


def find_references_csv(claims_file: str, references_path: str) -> Path:
    """
    Derive references filename from claims filename.

    Args:
        claims_file: Path to claims CSV file
        references_path: Path to references directory

    Returns:
        Path to references CSV file
    """
    claims_name = Path(claims_file).stem  # e.g., "geneknow_claims"
    # Remove "_claims" suffix and add "_references"
    base_name = claims_name.replace("_claims", "")
    ref_filename = f"{base_name}_references.csv"
    ref_path = Path(references_path) / ref_filename
    return ref_path


def check_papers_exist(references_df: pd.DataFrame, papers_path: str) -> None:
    """
    Check if all required paper files exist (in .md, .html, or .htm format).
    Exit with error if any missing. Skips references where real_reference is false.

    Args:
        references_df: DataFrame with references
        papers_path: Path to papers directory
    """
    missing = []
    supported_extensions = ['.md', '.html', '.htm']

    for idx in references_df.index:
        row = references_df.loc[idx]
        # Skip if not a real reference
        real_ref = str(row['real_reference']).lower(
        ) if 'real_reference' in row else ''
        if real_ref == 'false':
            continue

        hash_id = row['hash_id']
        if pd.isna(hash_id) or str(hash_id).strip() == "":
            continue

        hash_id_str = str(hash_id).strip()

        # Check if ANY supported format exists
        found = False
        for ext in supported_extensions:
            paper_file = Path(papers_path) / f"{hash_id_str}{ext}"
            if paper_file.exists():
                found = True
                break

        if not found:
            missing.append(f"{hash_id_str} (need .md/.html/.htm)")

    if missing:
        print(f"Error: Missing paper files for hash_ids: {', '.join(missing)}")
        sys.exit(1)


def get_claims_for_reference(claims_df: pd.DataFrame, citation,
                             answered_yes_ids: set) -> list:
    """
    Find claim indices that cite this reference and haven't been answered "yes" yet.

    Args:
        claims_df: DataFrame with claims
        citation: Reference citation like "[1]"
        answered_yes_ids: Set of claim IDs already answered "yes"

    Returns:
        List of tuples (df_index, claim_id) to process
    """
    result = []
    citation_str = str(citation)
    for idx, row in claims_df.iterrows():
        claim_id = row['id']  # Use actual claim ID from the id column
        if claim_id in answered_yes_ids:
            continue
        # Parse citation column: "[1];[2];[3]"
        citations = [c.strip() for c in str(row['citation']).split(';')]
        if citation_str in citations:
            result.append((idx, claim_id))  # Return both df index and claim_id
    return result


def answer_questions_batch(questions: list, passage: str, model: str,
                           client, tracker: TokenTracker) -> list:
    """
    Answer questions based on paper passage using LLM.

    Args:
        questions: List of dicts with "id" and "question"
        passage: Full text from paper
        model: LLM model name
        client: OpenAI client
        tracker: TokenTracker instance

    Returns:
        List of answers with "id", "answer", "explanation"
    """
    output_schema = {
        "title": "AnswerObject",
        "type": "array",
        "items": {
            "type": "object",
            "required": ["id", "answer", "explanation"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "integer"},
                "answer": {"type": "string", "enum": ["Yes", "No", "Idk"]},
                "explanation": {"type": "string"}
            }
        }
    }

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a question answering assistant. Use ONLY the provided passage to answer received questions. Do not use outside knowledge."
            },
            {
                "role": "developer",
                "content": """
Task: Answer each YES/NO question using ONLY the provided passage.

Decision policy:
- "Yes" if the passage explicitly states the asked fact, or the fact is an unavoidable direct implication.
- "No" if the passage explicitly states the opposite, or an unavoidable direct implication contradicts it.
- "Idk" if the passage does not contain enough information to decide "Yes" vs "No", even after reasonable inference.

Instructions:
- Read in the JSON formatted questions provided
- Answer questions only based on the input passage
- The only legal answers are Yes, No and Idk. Do not output anything other than Yes/No/Idk.
- For "Yes" answers: explanation must be empty string ""
- For "No" or "Idk" answers: provide brief explanation (max 15 words)
- Do not skip any question. Always output the same number of items as the input list.
- Return the answers with question id in a JSON array

Aliases:
Cbfa1=RUNX2
TTF-1=NXK2-1
NEUROD1=NEUROD
FOXO1=FKHR

Output format:
[
    {"id": <same>, "answer": "Yes", "explanation": ""},
    {"id": <same>, "answer": "No", "explanation": "brief explanation max 15 words"},
    {"id": <same>, "answer": "Idk", "explanation": "brief explanation max 15 words"}
]

Return [] if there's no question in the input.
"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Answer these questions based only on the provided passage:"
                    },
                    {
                        "type": "input_text",
                        "text": f"<questions>{json.dumps(questions)}</questions>"
                    },
                    {
                        "type": "input_text",
                        "text": f"<passage>{passage}</passage>"
                    }
                ]
            }
        ]
    )

    # Update token tracker
    tracker.update(response.usage)

    # Parse and validate
    structured_output = json.loads(response.output_text)
    validate(instance=structured_output, schema=output_schema)

    # Validate that we got the right number of answers
    if len(structured_output) != len(questions):
        raise Exception(
            f"Expected {len(questions)} answers, got {len(structured_output)}. "
            f"Question IDs: {[q['id'] for q in questions]}"
        )

    return structured_output


def verify_claims(claims_file: str, references_path: str, papers_path: str,
                  model: str = "gpt-5-mini") -> None:
    """
    Main verification logic.

    Args:
        claims_file: Path to claims CSV file
        references_path: Path to references directory
        papers_path: Path to papers directory
        model: LLM model to use
    """
    # Initialize
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    tracker = TokenTracker()

    # Load files
    print(f"Loading claims from: {claims_file}")
    claims_df = pd.read_csv(claims_file)

    ref_file = find_references_csv(claims_file, references_path)
    print(f"Loading references from: {ref_file}")
    if not ref_file.exists():
        print(f"Error: References file not found: {ref_file}")
        sys.exit(1)
    references_df = pd.read_csv(ref_file)

    # Check all papers exist
    print("Checking paper files...")
    check_papers_exist(references_df, papers_path)
    print(f"All {len(references_df)} paper files found")

    # Initialize new columns
    claims_df['paper_answer'] = ''
    claims_df['answer_note'] = ''

    # Track which claims have been answered "yes" (using claim IDs, not df indices)
    claims_answered_yes_ids = set()

    # Process each reference
    for idx in references_df.index:
        ref_row = references_df.loc[idx]
        citation = ref_row['in_text_citation']
        hash_id = ref_row['hash_id']

        # Skip if not a real reference
        real_ref = str(ref_row['real_reference']).lower(
        ) if 'real_reference' in ref_row else ''
        if real_ref == 'false':
            print(f"Skipping {citation} - not a real reference")
            continue

        # Skip if hash_id is empty
        if pd.isna(hash_id) or str(hash_id).strip() == "":
            print(f"Skipping {citation} - no paper file")
            continue

        # Find claims for this reference that haven't been answered "yes"
        claim_info = get_claims_for_reference(claims_df, citation,
                                              claims_answered_yes_ids)

        if not claim_info:
            print(f"Skipping {citation} - no remaining claims to verify")
            continue

        hash_id_str = str(hash_id).strip()
        print(
            f"\nProcessing {citation} ({hash_id_str}): {len(claim_info)} claims")

        # Extract full text from paper (tries .md first, then .html/.htm)
        try:
            full_text = extract_full_text_from_paper(hash_id_str, papers_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error extracting text from {hash_id_str}: {e}")
            sys.exit(1)

        # Calculate and display paper statistics
        num_paragraphs = len([p for p in full_text.split('\n\n') if p.strip()])
        num_words = len(full_text.split())
        print(f"  Paper text: {num_paragraphs} paragraphs, {num_words} words")

        # Prepare questions
        questions = []
        empty_question_ids = []
        for df_idx, claim_id in claim_info:
            question_text = claims_df.loc[df_idx, 'question']
            if pd.notna(question_text) and str(question_text).strip():
                questions.append({
                    "id": claim_id,  # Use claim_id, not df_idx
                    "question": str(question_text)
                })
            else:
                empty_question_ids.append(claim_id)

        # Warn about empty questions
        if empty_question_ids:
            print(
                f"  Warning: Skipping claims with empty questions: {empty_question_ids}")

        if not questions:
            print(f"  No valid questions for {citation}")
            continue

        # Answer questions
        try:
            answers = answer_questions_batch(questions, full_text, model,
                                             client, tracker)
        except Exception as e:
            print(f"Error answering questions for {citation}: {e}")
            sys.exit(1)

        # Update claims with answers
        for answer in answers:
            claim_id = answer['id']
            # Find the df_idx for this claim_id
            for df_idx, cid in claim_info:
                if cid == claim_id:
                    claims_df.loc[df_idx, 'paper_answer'] = answer['answer']
                    claims_df.loc[df_idx, 'answer_note'] = answer.get(
                        'explanation', '')
                    break

            # Track if answered "yes" (using claim_id)
            if answer['answer'] == 'Yes':
                claims_answered_yes_ids.add(claim_id)

    # Save results
    claims_df.to_csv(claims_file, index=False)
    print(f"\nSuccessfully updated: {claims_file}")
    print(f"Added columns: 'paper_answer', 'answer_note'")

    # Calculate and display final statistics
    total_claims = len(claims_df)
    yes_count = len(claims_answered_yes_ids)
    percentage = (yes_count / total_claims * 100) if total_claims > 0 else 0
    print(
        f"\nResults: {yes_count} out of {total_claims} claims verified Yes ({percentage:.1f}%)")

    # Print token report
    tracker.print_report()


def main():
    parser = argparse.ArgumentParser(
        description="Verify claims against paper content using LLM."
    )
    parser.add_argument(
        "claims_file",
        help="Path to claims CSV file"
    )
    parser.add_argument(
        "references_path",
        help="Path to references directory"
    )
    parser.add_argument(
        "papers_path",
        help="Path to papers directory"
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="LLM model to use (default: gpt-5-mini)"
    )

    args = parser.parse_args()

    verify_claims(args.claims_file, args.references_path,
                  args.papers_path, args.model)


if __name__ == "__main__":
    main()
