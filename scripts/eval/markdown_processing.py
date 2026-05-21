#!/usr/bin/env python3
"""
Extract text from Markdown files (e.g., converted academic papers).

This module reads Markdown files and extracts clean text, keeping only
the abstract, introduction, results, and discussion sections while
filtering out figures, tables, references, methods, and acknowledgments.

Can be used as a module or CLI script.

Usage as CLI:
    python markdown_processing.py --input <md_file>
    
Example:
    python markdown_processing.py --input ./article.md

Usage as module:
    from markdown_processing import extract_full_text_from_markdown
    
    full_text, metadata = extract_full_text_from_markdown(
        "/path/to/article.md",
        verbose=True
    )
    
    if metadata['success']:
        print(f"Extracted {metadata['total_words']} words")
"""

import argparse
import os
import re
import sys
from typing import Tuple, Dict, Optional


__all__ = [
    'extract_full_text_from_markdown',
    'extract_full_text_from_markdown_file',
    'detect_section_header',
    'is_section_header_line',
    'normalize_section_name',
    'should_skip_line',
    'is_figure_caption_start',
    'is_doi_or_url_line',
]


# Sections that indicate START of content (in order of preference)
START_SECTION_KEYWORDS = [
    'abstract', 'summary'
]

# Sections to KEEP (once we start)
KEEP_SECTION_KEYWORDS = [
    'abstract', 'summary', 'introduction', 'background',
    'results', 'discussion', 'conclusions'
]

# Sections to SKIP (stop processing when encountered)
SKIP_SECTION_KEYWORDS = [
    'methods', 'materials and methods', 'experimental procedures', 'methodology',
    'references', 'bibliography', 'literature cited',
    'acknowledgment', 'acknowledgement', 'funding', 'disclosure',
    'supplementary', 'appendix', 'data availability',
    'conflict of interest', 'competing interests',
    'author contributions', 'author information', 'authorship',
    'figure', 'figures', 'figure legends', 'tables',
    'supplemental information', 'supplementary materials', "Supplementary Information"
]

# Patterns to remove from content
REMOVE_PATTERNS = [
    # Table captions
    r'\*\*Table \d+\.\*\*.*?\n',
    r'Table \d+\..*?\n',
    # Page numbers and journal metadata
    r'^\s*\d+\s*$',  # Page numbers (standalone numbers)
    r'^\s*Volume \d+.*?\n',  # Journal volume lines
    r'^\s*Copyright ©.*?\n',  # Copyright lines
    r'^\s*Received:.*?\n',  # Received date lines
    r'^\s*Accepted:.*?\n',  # Accepted date lines
    r'^\s*Published online:.*?\n',  # Published date lines
    r'^\s*DOI:.*?\n',  # DOI lines
    r'^\s*https://doi\.org/.*?\n',  # DOI URLs
    r'^\s*ARTICLE\s*$',  # ARTICLE labels
    r'^\s*RESEARCH ARTICLE\s*$',  # RESEARCH ARTICLE labels
    r'^\s*Point-of-View\s*$',  # Point-of-View labels
    # Journal headers like "Development (2025) 152, dev204771. doi:..."
    r'^\s*Development \(\d{4}\) \d+,.*?doi:',
    r'^\s*NATURE COMMUNICATIONS.*?doi:',  # Nature Communications headers
    r'^\s*Cell, Vol\. \d+,',  # Cell journal headers
    r'^\s*Epigenetics\s*$',  # Journal section labels
    r'^\s*www\.[a-z]+\.(com|org).*?\n',  # Journal URLs
    r'^\s*Cell \d+\s*$',  # Cell page numbers
    r'^\s*PLOS ONE.*?\n',  # PLOS ONE headers
    r'^\s*\d+ / \d+\s*$',  # Page counts like "1 / 21"
]

# Pre-compute lowercase versions of keyword lists for exact matching
_START_SECTION_KEYWORDS_LOWER = [kw.lower() for kw in START_SECTION_KEYWORDS]
_SKIP_SECTION_KEYWORDS_LOWER = [kw.lower() for kw in SKIP_SECTION_KEYWORDS]
_KEEP_SECTION_KEYWORDS_LOWER = [kw.lower() for kw in KEEP_SECTION_KEYWORDS]

# Combined list of all section keywords for standalone detection
_ALL_SECTION_KEYWORDS_LOWER = (
    _START_SECTION_KEYWORDS_LOWER +
    _SKIP_SECTION_KEYWORDS_LOWER +
    _KEEP_SECTION_KEYWORDS_LOWER
)

# Regex pattern for decorative characters to strip from section names
# Includes: = - * _ # ~ + ! @ $ % ^ & ( ) [ ] { } | / \ : ; " ' , . ? *
# Note: Does NOT include whitespace - spaces are preserved for multi-word keywords
_DECORATIVE_CHARS_PATTERN = r'[=\-_#~+!@$%^&()\[\]{}|/\\:;"\'\,\.\?*]+'


def normalize_section_name(section_name: str) -> str:
    """
    Normalize section name for exact matching.
    Removes markdown syntax, decorative characters, and lowercases.
    Preserves spaces in multi-word keywords.
    """
    # Strip leading/trailing whitespace
    normalized = section_name.strip()

    # Remove leading markdown header markers (any number of # followed by space)
    normalized = re.sub(r'^#+\s*', '', normalized)

    # Strip again after removing header markers
    normalized = normalized.strip()

    # Replace all decorative characters with empty string
    # This removes them while keeping letters and spaces intact
    normalized = re.sub(_DECORATIVE_CHARS_PATTERN, '', normalized)

    # Lowercase and final strip
    normalized = normalized.lower().strip()

    return normalized


def is_section_header_line(line: str) -> bool:
    """
    Check if a line appears to be a section header.

    Two cases:
    1. Markdown headers: any number of # followed by space (e.g., #, ##, ###, ####)
    2. Standalone section keywords: line contains only a keyword (after normalization)
    """
    line_stripped = line.strip()

    if not line_stripped:
        return False

    # Case 1: Markdown header (any level: #, ##, ###, ####, etc.)
    if re.match(r'^#+\s', line_stripped):
        return True

    # Case 2: Standalone keyword (after removing decorative chars, must be exact match)
    normalized = normalize_section_name(line_stripped)
    if normalized in _ALL_SECTION_KEYWORDS_LOWER:
        return True

    return False


def detect_section_header(line: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Detect if a line is a section header using exact matching.

    Returns:
        Tuple of (section_type, section_name, is_header) where:
        - section_type: 'start', 'keep', 'skip', or None
        - section_name: normalized section name or None
        - is_header: True if the line is a header line (known or unknown)
    """
    line_stripped = line.strip()

    if not line_stripped:
        return None, None, False

    # Check if this is a markdown header or standalone potential header
    is_header = is_section_header_line(line)

    if not is_header:
        return None, None, False

    # Normalize for matching
    normalized = normalize_section_name(line_stripped)

    if not normalized:
        return None, None, True

    # Check for exact match in start sections
    if normalized in _START_SECTION_KEYWORDS_LOWER:
        return 'start', normalized, True

    # Check for exact match in skip sections
    if normalized in _SKIP_SECTION_KEYWORDS_LOWER:
        return 'skip', normalized, True

    # Unknown section (including keep sections) but is a header
    return None, normalized, True


def is_figure_caption_start(line: str) -> bool:
    """
    Check if a line starts a figure caption.
    Matches patterns like:
    - **Fig 1. Title**
    - **Figure 1. Title**
    - **Fig. 1. Title**
    - Figure 1. Title (without bold)
    - **Fig. 7** caption text
    - Fig. 7 | caption text
    """
    line_stripped = line.strip()
    # Bold figure captions (e.g., "**Fig. 7**" or "**Fig. 7."**)
    if re.match(r'^\*\*Fig(ure)?\.? \d+[A-Z]?\b', line_stripped, re.IGNORECASE):
        return True
    # Non-bold figure captions at start of line (e.g., "Fig. 7 |" or "Fig. 7.")
    if re.match(r'^Fig(ure)?\.? \d+[A-Z]?\b', line_stripped, re.IGNORECASE):
        return True
    return False


def is_doi_or_url_line(line: str) -> bool:
    """Check if line is a DOI or URL line that follows figure captions."""
    line_stripped = line.strip()
    if re.match(r'^https://doi\.org/', line_stripped):
        return True
    if re.match(r'^doi:', line_stripped, re.IGNORECASE):
        return True
    return False


def is_picture_placeholder(line: str) -> bool:
    """Detect picture placeholder lines like **==> picture [145 x 191] intentionally omitted <==**"""
    return '**==> picture' in line and 'intentionally omitted <==**' in line


def is_picture_text_start(line: str) -> bool:
    """Detect start of picture text block."""
    return '**----- Start of picture text -----' in line or '----- Start of picture text -----' in line


def is_picture_text_end(line: str) -> bool:
    """Detect end of picture text block."""
    return '**----- End of picture text -----' in line or '----- End of picture text -----' in line


def should_skip_line(line: str) -> bool:
    """
    Check if a line should be skipped (figures, tables, page numbers, etc.)
    """
    line_stripped = line.strip()

    if not line_stripped:
        return False  # Keep empty lines for now

    # Check against remove patterns
    for pattern in REMOVE_PATTERNS:
        if re.search(pattern, line, re.IGNORECASE | re.MULTILINE | re.DOTALL):
            return True

    # Check if it's a markdown table line
    if re.match(r'^\s*\|.*\|\s*$', line):
        return True

    # Check if it's a table separator line
    if re.match(r'^\s*\|[-:\s|]+\|\s*$', line):
        return True

    return False


def extract_full_text_from_markdown(md_text: str) -> str:
    """
    Extract main text from markdown, keeping only Abstract, Introduction, Results, and Discussion.

    Logic:
    1. Start from Abstract, Summary, or Introduction if found
    2. If not found, start from the beginning
    3. Keep content through Results and Discussion
    4. Stop at Methods, References, Acknowledgments, or anything after Discussion

    Args:
        md_text: Markdown content as string

    Returns:
        String containing extracted text
    """
    if not md_text or not md_text.strip():
        return ""

    lines = md_text.split('\n')
    result = []

    # State tracking
    started = False
    passed_discussion = False  # Have we passed through discussion?
    current_skip = False       # Currently inside a skip section
    in_table = False
    in_figure_caption = False
    in_picture_block = False

    # First pass: find if we have a start section
    has_start_section = False
    for line in lines:
        section_type, _, _ = detect_section_header(line)
        if section_type == 'start':
            has_start_section = True
            break

    # Second pass: extract content
    for i, line in enumerate(lines):
        # Check if this is a section header
        section_type, section_name, is_header = detect_section_header(line)

        if is_header:
            # If we were in a figure caption or picture block, section header ends it
            in_figure_caption = False
            in_picture_block = False

            # End current skip section if we were in one
            current_skip = False

            # Handle start logic
            if not started:
                if section_type == 'start':
                    started = True
                elif not has_start_section:
                    started = True

            # Stop if we passed discussion and hit next header
            if passed_discussion:
                break

            # Check if this is discussion (exact match)
            if section_name == 'discussion':
                passed_discussion = True
                if started:
                    result.append(line)  # Keep discussion header
                continue

            # Check if this is a skip section
            if section_type == 'skip':
                current_skip = True  # Skip entire section
                continue

            # Keep other headers
            if started:
                result.append(line)
            continue

        # Skip content if in skip section
        if current_skip:
            continue

        # If we haven't started yet and there's no start section, start now
        if not started and not has_start_section:
            started = True

        # Skip if we haven't started
        if not started:
            continue

        # Check for picture placeholder (standalone, no block)
        if is_picture_placeholder(line):
            continue

        # Check for picture text block start
        if is_picture_text_start(line):
            in_picture_block = True
            continue

        # If we're in a picture text block, check if it ends
        if in_picture_block:
            if is_picture_text_end(line):
                in_picture_block = False
            # Skip all lines in picture block (including end line)
            continue

        # Check for figure caption start
        if is_figure_caption_start(line):
            in_figure_caption = True
            continue

        # If we're in a figure caption, check if it continues or ends
        if in_figure_caption:
            line_stripped = line.strip()
            # Empty line ends the caption
            if not line_stripped:
                in_figure_caption = False
                continue
            # DOI/URL line ends the caption (these are metadata lines)
            if is_doi_or_url_line(line):
                in_figure_caption = False
                continue
            # Next figure or section ends caption (handled above for sections)
            if is_figure_caption_start(line):
                in_figure_caption = True
                continue
            # Otherwise, skip this line as it's part of the caption
            continue

        # Check if this line should be skipped
        if should_skip_line(line):
            continue

        # Check for table block
        if re.match(r'^\s*\|', line):
            in_table = True
            continue
        elif in_table and not line.strip():
            in_table = False
            continue
        elif in_table:
            continue

        # Keep the line
        result.append(line)

    # Join and clean
    text = '\n'.join(result)
    return clean_text(text)


def clean_text(text: str) -> str:
    """
    Clean up extracted text by removing artifacts and normalizing whitespace.
    """
    if not text:
        return ""

    # Remove remaining figure/table references that look like artifacts
    # But keep inline figure references like "(Fig. 1A)" or "see Figure 2"

    # Split into paragraphs (sequences of non-empty lines)
    paragraphs = []
    current_paragraph = []

    for line in text.split('\n'):
        if line.strip():
            current_paragraph.append(line)
        else:
            # Empty line - save current paragraph if it has content
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []

    # Don't forget the last paragraph
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))

    # Join paragraphs with single empty line between them
    text = '\n\n'.join(paragraphs)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def count_words(text: str) -> int:
    """
    Count words in text.
    """
    if not text:
        return 0
    # Split on whitespace and filter empty strings
    words = [w for w in text.split() if w.strip()]
    return len(words)


def extract_full_text_from_markdown_file(md_file_path: str, verbose: bool = True) -> Tuple[str, Dict]:
    """
    Extract text from a Markdown file.

    Args:
        md_file_path: Path to the Markdown file to process
        verbose: Print progress messages to stderr (default: True)

    Returns:
        tuple: (full_text_string, metadata_dict)
        - full_text_string: Extracted text (empty string if error)
        - metadata_dict: {
            'file_path': str,
            'success': bool,
            'error': str or None,
            'num_paragraphs': int,
            'total_words': int
        }
    """
    metadata = {
        'file_path': md_file_path,
        'success': False,
        'error': None,
        'num_paragraphs': 0,
        'total_words': 0
    }

    try:
        if verbose:
            print(f"Reading Markdown file: {md_file_path}", file=sys.stderr)

        # Check if file exists
        if not os.path.exists(md_file_path):
            metadata['error'] = f"File not found: {md_file_path}"
            if verbose:
                print(f"Error: {metadata['error']}", file=sys.stderr)
            return "", metadata

        # Read Markdown content from file
        with open(md_file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        if verbose:
            print(
                f"File loaded: {len(md_content)} characters", file=sys.stderr)

        # Extract full text
        full_text = extract_full_text_from_markdown(md_content)

        if not full_text:
            metadata['error'] = "No text extracted from the content"
            if verbose:
                print(f"Warning: {metadata['error']}", file=sys.stderr)
            return "", metadata

        # Calculate metadata
        # Count paragraphs by splitting on double newlines
        paragraphs = [p for p in full_text.split('\n\n') if p.strip()]
        metadata['num_paragraphs'] = len(paragraphs)
        metadata['total_words'] = count_words(full_text)
        metadata['success'] = True

        if verbose:
            print(f"Extracted {metadata['num_paragraphs']} paragraph(s), {metadata['total_words']} words",
                  file=sys.stderr)

        return full_text, metadata

    except UnicodeDecodeError as e:
        metadata['error'] = f"Could not read file (encoding issue): {e}"
        if verbose:
            print(f"Error: {metadata['error']}", file=sys.stderr)
        return "", metadata

    except ValueError as e:
        metadata['error'] = str(e)
        if verbose:
            print(f"Error: {e}", file=sys.stderr)
        return "", metadata

    except Exception as e:
        metadata['error'] = f"Unexpected error: {e}"
        if verbose:
            print(f"Unexpected Error: {e}", file=sys.stderr)
        return "", metadata


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Extract main text from a Markdown file.'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to the Markdown file to process'
    )

    args = parser.parse_args()

    # Extract text
    full_text, metadata = extract_full_text_from_markdown_file(
        args.input,
        verbose=True
    )

    # Check success
    if not metadata['success']:
        sys.exit(1)

    # Print extracted text to stdout
    print(full_text)

    # Print summary to stderr
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Summary: {metadata['num_paragraphs']} paragraphs, {metadata['total_words']} total words",
          file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == '__main__':
    main()
