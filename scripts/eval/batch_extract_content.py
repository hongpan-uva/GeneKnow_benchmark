#!/usr/bin/env python3
"""
Batch extract main content from Markdown and HTML files.

This script processes all .md, .htm, and .html files in a specified folder,
extracts the main content from each, and displays a compact summary.

Usage:
    python batch_extract_content.py --input /path/to/folder
    
Example:
    python batch_extract_content.py --input ./papers
"""

import argparse
import os
import sys

# Import the processing modules
import markdown_processing
import html_processing


def get_first_n_words(text: str, n: int = 10) -> str:
    """Return first n words from text."""
    if not text:
        return ""
    words = text.split()[:n]
    return ' '.join(words)


def get_last_n_words(text: str, n: int = 10) -> str:
    """Return last n words from text."""
    if not text:
        return ""
    words = text.split()[-n:]
    return ' '.join(words)


def format_content_preview(text: str, first_n: int = 10, last_n: int = 10) -> str:
    """Format content as 'first10...last10'."""
    if not text or not text.strip():
        return "[empty]"
    
    words = text.split()
    total_words = len(words)
    
    if total_words <= first_n + last_n:
        # If text is short, show all of it
        return ' '.join(words)
    
    first = get_first_n_words(text, first_n)
    last = get_last_n_words(text, last_n)
    return f"{first}...{last}"


def scan_folder(folder_path: str) -> list:
    """
    Scan folder for .md, .htm, and .html files.
    
    Returns:
        List of tuples: (filename, full_path, file_type)
    """
    files = []
    
    if not os.path.isdir(folder_path):
        print(f"ERROR: '{folder_path}' is not a valid directory", file=sys.stderr)
        return files
    
    for filename in os.listdir(folder_path):
        full_path = os.path.join(folder_path, filename)
        
        # Skip directories and non-files
        if not os.path.isfile(full_path):
            continue
        
        # Check extension (case-insensitive)
        lower_name = filename.lower()
        if lower_name.endswith('.md'):
            files.append((filename, full_path, 'markdown'))
        elif lower_name.endswith('.htm') or lower_name.endswith('.html'):
            files.append((filename, full_path, 'html'))
    
    # Sort by filename for consistent output
    files.sort(key=lambda x: x[0])
    
    return files


def process_file(filename: str, full_path: str, file_type: str) -> tuple:
    """
    Process a single file and extract content.
    
    Returns:
        Tuple: (success: bool, preview: str, paragraphs: str, words: str, error: str)
    """
    try:
        if file_type == 'markdown':
            text, metadata = markdown_processing.extract_full_text_from_markdown_file(
                full_path, verbose=False
            )
        else:  # html
            text, metadata = html_processing.extract_full_text_from_html_file(
                full_path, verbose=False
            )
        
        if metadata.get('success', False):
            preview = format_content_preview(text)
            paragraphs = str(metadata.get('num_paragraphs', 0))
            words = str(metadata.get('total_words', 0))
            return (True, preview, paragraphs, words, None)
        else:
            error_msg = metadata.get('error', 'Unknown error')
            return (False, f"ERROR: {error_msg}", "-", "-", error_msg)
            
    except Exception as e:
        return (False, f"ERROR: {str(e)}", "-", "-", str(e))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch extract main content from Markdown and HTML files.'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to the folder containing .md, .htm, and .html files'
    )
    
    args = parser.parse_args()
    folder_path = args.input
    
    # Scan folder for files
    files = scan_folder(folder_path)
    
    if not files:
        print(f"No .md, .htm, or .html files found in '{folder_path}'")
        sys.exit(0)
    
    print(f"Found {len(files)} file(s) to process in '{folder_path}'\n")
    
    # Process each file
    successful = 0
    failed = 0
    
    for filename, full_path, file_type in files:
        success, preview, paragraphs, words, error = process_file(
            filename, full_path, file_type
        )
        
        # Print compact output (filename | stats | content)
        print(f"{filename} | {paragraphs} paras | {words} words | {preview}")
        print()
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Total: {len(files)} files | Successful: {successful} | Failed: {failed}")
    print("=" * 60)
    
    # Exit with error code if any files failed
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
