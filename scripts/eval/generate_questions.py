#!/usr/bin/env python3
"""
Verify claims by generating YES/NO questions from claim text.

Usage:
    python verify_claim.py <claims_file> [--model <model_name>]
    
Example:
    python verify_claim.py ../discover_cases/case1/claims/geneknow_claims.csv
    python verify_claim.py ../discover_cases/case1/claims/geneknow_claims.csv --model gpt-5
"""

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from openai import OpenAI
from jsonschema import validate
from token_tracker import TokenTracker


def generate_questions_batch(claims: list, model: str, client, tracker: TokenTracker) -> list:
    """
    Generate YES/NO questions for all claims at once using a single API call.

    Args:
        claims: List of claim strings
        model: The LLM model to use
        client: OpenAI client instance
        tracker: TokenTracker instance

    Returns:
        List of questions (same order as input claims)
    """
    # Format claims as JSON array with IDs
    claims_json = [{"id": i+1, "claim": claim}
                   for i, claim in enumerate(claims)]

    output_schema = {
        "title": "QuestionsArray",
        "type": "array",
        "items": {
            "type": "object",
            "required": ["id", "question"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "integer"},
                "question": {"type": "string"}
            }
        }
    }

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a question generator. Generate exactly one YES/NO question for each claim provided."
            },
            {
                "role": "developer",
                "content": """
Definition of "YES/NO question":
- The question must be answerable strictly with "yes" or "no" without requiring any extra information.
- It should be a single question (no "and/or", no two-part questions).
- Avoid open-ended forms like "explain", "describe", "why", "how".

Instructions:
- Read the JSON array of claims provided by the user.
- Generate exactly ONE YES/NO question for each claim.
- Each question should be answerable as "Yes" based on the corresponding claim text.
- Keep each question concise (ideally ≤ 20 words).
- If a claim is ambiguous or missing key context such that a yes/no answer would be meaningless, output "" for that claim.
- If a claim is purely subjective/opinion with no factual core, output "" for that claim.

Return a JSON array with the same number of items as the input, maintaining the same order:
[
    {"id": 1, "question": "<question for claim 1 or empty string>"},
    {"id": 2, "question": "<question for claim 2 or empty string>"},
    ...
]

Input format:
[
    {"id": 1, "claim": "claim text 1"},
    {"id": 2, "claim": "claim text 2"},
    ...
]
"""
            },
            {
                "role": "user",
                "content": f"Generate YES/NO questions for the following claims:\n\n{json.dumps(claims_json, indent=2)}"
            }
        ]
    )

    # Update token tracker with this API call's usage
    tracker.update(response.usage)

    # Parse JSON response
    structured_output = json.loads(response.output_text)
    validate(instance=structured_output, schema=output_schema)

    # Validate that we got the right number of questions
    if len(structured_output) != len(claims):
        raise Exception(
            f"Expected {len(claims)} questions, got {len(structured_output)}")

    # Validate that IDs match and extract questions in order
    questions = []
    for i, item in enumerate(structured_output):
        if item["id"] != i + 1:
            raise Exception(
                f"ID mismatch at position {i}: expected {i+1}, got {item['id']}")
        questions.append(item["question"])

    return questions


def generate_questions(claims_file: str, model: str = "gpt-5-mini", tracker=None) -> None:
    """
    Process claims file: generate questions and add claim_answer column.

    Args:
        claims_file: Path to the claims CSV file
        model: LLM model to use for question generation
        tracker: Optional TokenTracker instance to accumulate token usage
    """
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Read claims file
    claims_path = Path(claims_file)
    if not claims_path.exists():
        print(f"Error: Claims file not found: {claims_file}")
        sys.exit(1)

    print(f"Reading claims from: {claims_file}")
    df = pd.read_csv(claims_file)

    # Check required columns
    required_cols = ['id', 'claim']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Found {len(df)} claims to process")
    print(f"Using model: {model}")

    # Initialize token tracker if not provided
    external_tracker = tracker is not None
    if not external_tracker:
        tracker = TokenTracker()

    # Extract all claims for batch processing
    all_claims = df['claim'].astype(str).tolist()

    # Generate all questions in ONE API call
    print(
        f"Generating questions for all {len(all_claims)} claims in single API call...", end=" ", flush=True)

    try:
        questions = generate_questions_batch(
            all_claims, model, client, tracker)
        print("Done")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Add new columns
    df['question'] = questions
    df['claim_answer'] = 'Yes'  # Always Yes since question is derived from claim

    # Save back to original file
    df.to_csv(claims_file, index=False)
    print(f"\nSuccessfully updated: {claims_file}")
    print(f"Added columns: 'question', 'claim_answer'")
    print(f"Total claims processed: {len(df)}")

    # Print token usage report only if using internal tracker
    if not external_tracker:
        tracker.print_report()

    # Check for empty questions and display warning
    empty_indices = [i for i, q in enumerate(
        questions) if not q or q.strip() == ""]
    if empty_indices:
        empty_ids = [str(df.iloc[i]['id']) for i in empty_indices]
        print(
            f"\n⚠️  Warning: {len(empty_ids)} claim(s) did not generate a question:")
        print(f"   Claim IDs: {', '.join(empty_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate YES/NO questions from claims and verify with Yes answers."
    )
    parser.add_argument(
        "claims_file",
        help="Path to the claims CSV file (e.g., ../discover_cases/case1/claims/geneknow_claims.csv)"
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="LLM model to use for question generation (default: gpt-5-mini)"
    )

    args = parser.parse_args()

    generate_questions(args.claims_file, args.model)


if __name__ == "__main__":
    main()
