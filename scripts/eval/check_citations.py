#!/usr/bin/env python3
"""
Check that all in-text citations in claim files have matching references.

This script validates that citations in claim files exist in the corresponding
reference files. It processes all claim files and reports any missing citations.

Usage:
    python check_citations.py --claim_path <path> --ref_path <path>

Exit codes:
    0 - All citations matched
    1 - One or more citations missing
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_citations(citation_cell):
    """
    Parse semicolon-separated citations from a cell.
    
    Args:
        citation_cell: String containing citations like "[1];[2];[3]"
    
    Returns:
        List of individual citation strings, empty list if cell is empty
    """
    if not citation_cell or citation_cell.strip() == '' or citation_cell.strip().lower() == 'nan':
        return []
    return [c.strip() for c in citation_cell.split(';') if c.strip()]


def read_csv_file(filepath):
    """Read a CSV file and return list of dictionaries."""
    rows = []
    with open(filepath, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def check_file(claim_file, ref_file):
    """
    Check one claim file against its reference file.
    
    Args:
        claim_file: Path to claim CSV file
        ref_file: Path to reference CSV file
    
    Returns:
        List of tuples (claim_id, citation) for missing citations
    """
    missing = []
    
    # Read files
    claims = read_csv_file(claim_file)
    refs = read_csv_file(ref_file)
    
    # Build set of reference citations for O(1) lookup
    ref_citations = set()
    for row in refs:
        citation = str(row.get('in_text_citation', '')).strip()
        if citation:
            ref_citations.add(citation)
    
    # Check each claim row
    for row in claims:
        claim_id = row.get('id', 'unknown')
        citation_cell = str(row.get('citation', '')).strip()
        
        # Parse citations
        citations = parse_citations(citation_cell)
        
        # Check each citation
        for citation in citations:
            if citation not in ref_citations:
                missing.append((claim_id, citation))
    
    return missing


def main():
    parser = argparse.ArgumentParser(
        description='Check that all citations in claim files have matching references.'
    )
    parser.add_argument(
        '--claim_path',
        required=True,
        help='Path to directory containing claim CSV files'
    )
    parser.add_argument(
        '--ref_path',
        required=True,
        help='Path to directory containing reference CSV files'
    )
    
    args = parser.parse_args()
    
    # Get all claim files
    claim_files = sorted(Path(args.claim_path).glob('*_claims.csv'))
    
    if not claim_files:
        print(f"No claim files found in {args.claim_path}", file=sys.stderr)
        sys.exit(1)
    
    # Track missing citations across all files
    all_missing = {}  # {claim_filename: [(claim_id, citation), ...]}
    files_checked = 0
    
    for claim_file in claim_files:
        # Construct corresponding reference filename
        claim_filename = claim_file.name
        ref_filename = claim_filename.replace('_claims.csv', '_references.csv')
        ref_file = Path(args.ref_path) / ref_filename
        
        if not ref_file.exists():
            print(f"Error: Reference file not found: {ref_file}", file=sys.stderr)
            all_missing[claim_filename] = [('N/A', f'Reference file missing: {ref_filename}')]
            continue
        
        # Check this file
        missing = check_file(str(claim_file), str(ref_file))
        files_checked += 1
        
        if missing:
            all_missing[claim_filename] = missing
    
    # Report results
    if all_missing:
        print("=" * 60)
        print("MISSING CITATIONS REPORT")
        print("=" * 60)
        
        total_missing = 0
        for claim_filename, missing_list in all_missing.items():
            print(f"\nFile: {claim_filename}")
            print("-" * 40)
            for claim_id, citation in missing_list:
                print(f"  Claim ID: {claim_id}, Missing citation: {citation}")
                total_missing += 1
        
        print("\n" + "=" * 60)
        print(f"SUMMARY: {total_missing} missing citation(s) across {len(all_missing)} file(s)")
        print(f"Files checked: {files_checked}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("=" * 60)
        print("✓ All citations matched successfully!")
        print(f"Files checked: {files_checked}")
        print("=" * 60)
        sys.exit(0)


if __name__ == '__main__':
    main()
