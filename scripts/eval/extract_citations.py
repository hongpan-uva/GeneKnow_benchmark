#!/usr/bin/env python3
"""
Extract in-text citations and their corresponding references from academic text files.

This script reads files containing synopsis with in-text citations and reference lists,
uses OpenAI's GPT-5-mini model to extract citation-reference mappings, and outputs
a CSV file with the results.

Usage:
    python extract_citations.py --input <input_file> --output_dir <output_directory>

Example:
    python extract_citations.py --input discover_out/GeneKnow.txt --output_dir ./results
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from jsonschema import validate, ValidationError
from openai import OpenAI


# JSON Schema for citation validation
CITATION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "in_text_citation": {"type": "string"},
            "reference": {"type": "string"}
        },
        "required": ["in_text_citation", "reference"]
    }
}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract in-text citations and references from academic text files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file containing synopsis and references",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output CSV file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="OpenAI model to use (default: gpt-5-mini)",
    )
    return parser.parse_args()


def validate_environment() -> str:
    """Check if OPENAI_API_KEY is set in environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it before running this script."
        )
    return api_key


def read_input_file(file_path: str) -> str:
    """Read the entire content of the input file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_citations_with_llm(
    text: str, client: OpenAI, model: str
) -> Tuple[List[Dict[str, str]], str]:
    """
    Use OpenAI LLM to extract in-text citations and their corresponding references.

    Args:
        text: The full text containing synopsis and references
        client: OpenAI client instance
        model: Model name to use

    Returns:
        Tuple of (citations, warning_message)
        - citations: List of dictionaries with 'in_text_citation' and 'reference' keys
        - warning_message: Validation warning if any, empty string otherwise
    """
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Extract in-text citations and match with full references. "
                    "Split citation ranges into individual entries."
                ),
            },
            {
                "role": "developer",
                "content": """
Extract citations from the text and return as JSON array.

Instructions:
1. Identify all in-text citation formats: e.g. [1], [2], (1), [1,2], [1-3], (Smith et al., 2020)
2. CRITICAL: Split citation ranges like [1-3] into INDIVIDUAL entries: [1], [2], [3]
   - Each number gets its own row with the corresponding reference
   - [3,4] becomes [3] and [4] as separate entries
3. Match each citation to its full reference from the reference list
4. Return JSON array ordered by first appearance:
[
    {"in_text_citation": "[1]", "reference": "Full reference text"},
    {"in_text_citation": "[2]", "reference": "Full reference text"}
]

Important:
- Split ALL ranges into individual citation entries
- Order by first appearance, remove duplicates (keep first occurrence)
- Output references exactly as what they are, do not do any edit
- Use "NOT_FOUND" if reference missing
- Return [] if no citations found
""",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract citations and split all ranges into individual entries:",
                    },
                    {"type": "input_text", "text": text},
                ],
            },
        ],
    )

    # Parse the JSON response
    try:
        citations = json.loads(response.output_text)
        if not isinstance(citations, list):
            return [], "Warning: Response is not a list"
    except json.JSONDecodeError as e:
        return [], f"Error parsing JSON response: {e}. Raw response: {response.output_text}"

    # Validate citations
    is_valid, valid_citations, warning = validate_citations(citations)

    return valid_citations, warning


def extract_url(text: str) -> str:
    """Extract the first http:// or https:// URL from text."""
    if not text:
        return ""
    pattern = r'https?://[^\s,;<>"\[\]]+'
    match = re.search(pattern, str(text))
    return match.group(0) if match else ""


def validate_citations(citations: List[Dict[str, str]]) -> Tuple[bool, List[Dict[str, str]], str]:
    """
    Validate citations against JSON schema and check for duplicates.

    Args:
        citations: List of citation dictionaries

    Returns:
        Tuple of (is_valid, valid_citations, warning_message)
        - is_valid: True if no duplicates found, False otherwise
        - valid_citations: List of citations that passed schema validation
        - warning_message: Description of validation issues (empty if no issues)
    """
    if not citations:
        return True, [], ""

    # Validate against JSON schema
    try:
        validate(instance=citations, schema=CITATION_SCHEMA)
    except ValidationError as e:
        return False, citations, f"JSON schema validation failed: {e.message}"

    # Check for duplicate in_text_citations
    seen_citations = set()
    duplicates = []

    for citation in citations:
        in_text = citation.get("in_text_citation", "")
        if in_text in seen_citations:
            duplicates.append(in_text)
        else:
            seen_citations.add(in_text)

    if duplicates:
        warning_msg = f"Duplicate citations found and will be kept (first occurrence): {duplicates}"
        return False, citations, warning_msg

    return True, citations, ""


def create_dataframe(citations: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Create a pandas DataFrame from the citation list.

    Args:
        citations: List of dictionaries with 'in_text_citation' and 'reference'

    Returns:
        DataFrame with columns: 'in_text_citation', 'reference', 'url', 
        'real_reference', 'real_url', 'alternative_url'
    """
    columns = ["in_text_citation", "reference", "url",
               "real_reference", "real_url", "alternative_url"]

    if not citations:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(citations)

    # Extract URLs from references
    df["url"] = df["reference"].apply(extract_url)

    # Add blank columns for manual annotation
    df["real_reference"] = ""
    df["real_url"] = ""
    df["alternative_url"] = ""

    # Ensure correct column order
    df = df[columns]

    return df


def save_to_csv(df: pd.DataFrame, output_dir: str, input_filename: str) -> str:
    """
    Save the DataFrame to a CSV file in the specified directory.

    Args:
        df: DataFrame to save
        output_dir: Directory path for output
        input_filename: Original input filename (used to generate output name)

    Returns:
        Path to the saved CSV file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate output filename based on input filename
    base_name = Path(input_filename).stem
    output_file = output_path / f"{base_name}_references.csv"

    df.to_csv(output_file, index=False, encoding="utf-8")
    return str(output_file)


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    args = parse_arguments()

    # Validate environment
    try:
        api_key = validate_environment()
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Read input file
    try:
        print(f"Reading input file: {args.input}")
        text_content = read_input_file(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract citations using LLM
    print(f"Extracting citations using model: {args.model}")
    citations, validation_warning = extract_citations_with_llm(
        text_content, client, args.model)

    # Print validation warning if any
    if validation_warning:
        print(f"Warning: {validation_warning}")
        print(f"Saving partial results ({len(citations)} citations)")

    if not citations:
        print("Warning: No citations were extracted from the text")
    else:
        print(f"Extracted {len(citations)} citation(s)")

    # Create DataFrame
    df = create_dataframe(citations)

    # Save to CSV
    output_file = save_to_csv(df, args.output_dir, args.input)
    print(f"Results saved to: {output_file}")

    # Print summary
    print("\nSummary:")
    print(f"  Total citations extracted: {len(df)}")
    print(f"  Output file: {output_file}")


if __name__ == "__main__":
    main()
