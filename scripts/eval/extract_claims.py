#!/usr/bin/env python3
"""
Extract atomic, checkable claims from synopsis files with their supporting citations.

Usage:
    python extract_claims.py --input <input_file> --output_dir <output_dir>
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from jsonschema import validate, ValidationError
from openai import OpenAI

# JSON Schema for claim validation
CLAIM_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "citation": {"type": "string"}
        },
        "required": ["claim", "citation"]
    }
}


def read_input_file(file_path: str) -> str:
    """Read the content of the input file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_claims_with_citations(text: str, client: OpenAI, model: str = "gpt-5-mini") -> list:
    """
    Extract atomic claims with their citations from the text using OpenAI API.

    Args:
        text: The input text containing synopsis and references
        client: OpenAI client instance
        model: The model to use for extraction

    Returns:
        List of dictionaries with 'claim' and 'citation' keys
    """
    system_prompt = "You are a claim extractor. Extract atomic, checkable claims from scientific synopsis text."

    developer_prompt = """
A claim is defined as one proposition anchored to a single subject-predicate relationship.

Instructions:
- Extract atomic, checkable claims that exactly reflect the source—no external facts, inference, or speculation.
- Preserve hedging/negation/conditions, entity names/aliases as written, and any numbers, units, directions, or temporal cues.
- Replace all pronouns and demonstratives such as "this population", "these cells" with the full explicit entity name described in the source text.
- Split long or compound sentences into multiple claims.
- Each claim must be self-contained and interpretable in isolation, restate necessary constraints, context, and entity names explicitly
- Use the fewest non-overlapping claims needed to fully cover the text.
- In the 'citation' field, output the in-text citation(s) that support each claim, or an empty string if a claim is not followed by any citation.

IMPORTANT - Citation Format:
- a variety of in-text citation formats could exist: e.g. [1], [2], (1), [1,2], [1-3], (Smith et al., 2020)
- Multiple citations for one claim must be listed side-by-side, separated by semi-colon without space: [1-3] must become [1];[2];[3], (Smith et al., 2020; Zhou et al., 2018) must become (Smith et al., 2020);(Zhou et al., 2018)
- Do not change the citation format in output, use the exact same in-text citations

Return the extracted claims in a JSON array format:
[
    {"claim": "claim text 1", "citation": "citation 1"},
    {"claim": "claim text 2", "citation": "citation 2"}
]

Return an empty array [] if no claims are found or the text is blank."""

    user_prompt = f"""Extract atomic claims with their citations from the following text.

Text to analyze:
{text}
"""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "developer", "content": developer_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.output_text


def parse_and_validate_claims(response_text: str) -> list:
    """
    Parse JSON response and validate against schema.

    Args:
        response_text: The JSON text from the API response

    Returns:
        Validated list of claim dictionaries

    Raises:
        ValueError: If JSON is invalid or doesn't match schema
    """
    # Extract JSON from response (in case there's extra text)
    try:
        # Try to find JSON array in the response
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group()

        claims = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")

    # Validate against schema
    try:
        validate(instance=claims, schema=CLAIM_SCHEMA)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e.message}")

    return claims


def add_sequential_ids(claims: list) -> list:
    """
    Add sequential string IDs to each claim.

    Args:
        claims: List of claim dictionaries

    Returns:
        List with 'id' key added to each claim (as string)
    """
    for i, claim in enumerate(claims, start=1):
        claim['id'] = str(i)
    return claims


def save_to_csv(claims: list, output_path: str):
    """
    Save claims to CSV file.

    Args:
        claims: List of claim dictionaries
        output_path: Path to output CSV file
    """
    df = pd.DataFrame(claims)
    # Reorder columns to put id first
    if 'id' in df.columns:
        df = df[['id', 'claim', 'citation']]

    df.to_csv(output_path, index=False)
    print(f"Saved {len(claims)} claims to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract atomic claims with citations from synopsis files"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input file containing synopsis and references'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Path to output directory for CSV file'
    )
    parser.add_argument(
        '--model',
        default='gpt-5-mini',
        help='OpenAI model to use (default: gpt-5-mini)'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize OpenAI client
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=api_key)

    # Read input file
    print(f"Reading input file: {args.input}")
    text = read_input_file(args.input)

    # Extract claims using OpenAI
    print("Extracting claims with citations...")
    response_text = extract_claims_with_citations(
        text, client, model=args.model)

    # Parse and validate claims
    print("Validating extracted claims...")
    claims = parse_and_validate_claims(response_text)

    # Add sequential IDs
    claims = add_sequential_ids(claims)

    # Generate output filename
    input_name = Path(args.input).stem
    output_file = os.path.join(args.output_dir, f"{input_name}_claims.csv")

    # Save to CSV
    save_to_csv(claims, output_file)

    print(f"Done! Extracted {len(claims)} claims.")


if __name__ == "__main__":
    main()
