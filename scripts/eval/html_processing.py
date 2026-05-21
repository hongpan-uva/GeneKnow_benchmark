#!/usr/bin/env python3
"""
Extract text paragraphs from locally saved HTML files.

This module reads HTML files (e.g., manually downloaded academic papers) 
and extracts clean text paragraphs, filtering out references, methods, 
and acknowledgments sections.

Can be used as a module or CLI script.

Usage as CLI:
    python html_processing.py --input <html_file>
    
Example:
    python html_processing.py --input ./article.html

Usage as module:
    from html_processing import extract_from_file
    
    paragraphs, metadata = extract_from_file(
        "/path/to/article.html",
        min_words=20,
        max_words=500,
        verbose=True
    )
    
    if metadata['success']:
        print(f"Got {metadata['num_paragraphs']} paragraphs")

Note:
    This module processes locally saved HTML files. It does not download content.
"""

import argparse
import os
import re
import sys

from bs4 import BeautifulSoup

from geneknow.processing.text import count_words, _split_into_chunks


__all__ = [
    'extract_from_file',
    'extract_paragraphs_from_html',
    'should_skip_section',
    'clean_html',
    'find_main_content',
    'extract_full_text_from_html',
    'extract_full_text_from_html_file'
]


# Section headers to skip (case-insensitive partial matching)
SKIP_SECTION_KEYWORDS = [
    'reference', 'bibliography', 'literature cited',
    'method', 'materials and methods', 'experimental procedures', 'methodology',
    'acknowledgment', 'acknowledgement', 'funding', 'disclosure',
    'supplementary', 'appendix', 'data availability', 'conflict of interest',
    'author contributions', 'supplemental information'
]


def should_skip_section(header_text):
    """
    Check if a section should be skipped based on its header text.

    Args:
        header_text: The text content of a section header (h1, h2, h3, etc.)

    Returns:
        True if the section should be skipped, False otherwise
    """
    if not header_text:
        return False

    header_lower = header_text.lower().strip()
    return any(keyword in header_lower for keyword in SKIP_SECTION_KEYWORDS)


def clean_html(html_text):
    """
    Clean HTML by removing scripts, styles, and other non-content elements.

    Args:
        html_text: Raw HTML string

    Returns:
        BeautifulSoup object with cleaned HTML
    """
    soup = BeautifulSoup(html_text, 'lxml')

    # Remove script and style elements
    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        element.decompose()

    return soup


def find_main_content(soup):
    """
    Find the main content area of an HTML document.

    Prioritizes semantic HTML5 tags and common content containers.

    Args:
        soup: BeautifulSoup object

    Returns:
        BeautifulSoup Tag object representing main content area
    """
    # Priority 1: <article> tag (semantic HTML5)
    article = soup.find('article')
    if article:
        return article

    # Priority 2: <main> tag
    main = soup.find('main')
    if main:
        return main

    # Priority 3: Common content divs
    content_divs = soup.find_all(['div', 'section'], {
        'id': re.compile(r'content|article|main', re.I),
        'class': re.compile(r'content|article|main|body', re.I)
    })
    if content_divs:
        # Return the largest one by text length
        return max(content_divs, key=lambda x: len(x.get_text()))

    # Fallback: body
    return soup.find('body') or soup


def extract_paragraphs_from_html(html_text, upper_limit=500, lower_limit=20):
    """
    Extract clean text paragraphs from HTML content.

    Similar to extract_paragraphs_from_xml() but for HTML documents.
    Filters out references, methods, and acknowledgments sections.

    Args:
        html_text: HTML content as string
        upper_limit: Maximum word count per paragraph (default: 500)
        lower_limit: Minimum word count per paragraph (default: 20)

    Returns:
        List of paragraph strings
    """
    # Clean and parse HTML
    soup = clean_html(html_text)

    # Find main content area
    main_content = find_main_content(soup)

    if not main_content:
        raise ValueError("Could not identify main content area in HTML")

    paragraphs = []
    skip_current_section = False

    # Iterate through all elements in main content
    for element in main_content.descendants:
        # Check if this is a header element
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            header_text = element.get_text(strip=True)
            skip_current_section = should_skip_section(header_text)
            continue

        # Skip if we're in a section that should be filtered
        if skip_current_section:
            continue

        # Process paragraph-like elements
        if element.name in ['p', 'div', 'section']:
            # Skip if inside figures/tables (similar to XML version)
            if element.find_parent(['figure', 'figcaption', 'table']):
                continue

            # Skip elements that contain other paragraph containers
            # (we only want leaf nodes to avoid duplication)
            has_paragraph_child = False
            for child in element.find_all(['p', 'div', 'section'], recursive=False):
                has_paragraph_child = True
                break

            if has_paragraph_child:
                continue

            # Extract text
            text = element.get_text(' ', strip=True)
            if not text:
                continue

            # Check word count
            word_count = count_words(text)

            if word_count < lower_limit:
                # Too short, skip
                continue
            elif word_count <= upper_limit:
                # Good size, keep it
                paragraphs.append(text)
            else:
                # Too long, chunk it
                soft_cap = 1.1
                target_n_chunks = word_count // upper_limit + 1
                target_chunk_size = word_count // target_n_chunks
                max_chunk_size = int(target_chunk_size * soft_cap)

                chunks = _split_into_chunks(text, max_chunk_size)
                paragraphs.extend(chunks)

    return paragraphs


def extract_full_text_from_html(html_text):
    """
    Extract complete main text from HTML, starting from Introduction through Discussion.

    Finds the "Introduction" section and starts extraction from there.
    Continues through Discussion section, then stops at end markers
    (Methods, Materials and methods, References, Acknowledgements, Additional information).
    Does not filter by word count or chunk long paragraphs.

    Args:
        html_text: HTML content as string

    Returns:
        String containing full text with paragraphs separated by newlines
    """
    # Clean and parse HTML
    soup = clean_html(html_text)

    # Find main content area
    main_content = find_main_content(soup)

    if not main_content:
        raise ValueError("Could not identify main content area in HTML")

    paragraphs = []

    # State tracking
    BEFORE_INTRO = 0
    IN_MAIN_TEXT = 1
    IN_DISCUSSION = 2
    AFTER_DISCUSSION = 3  # Stop extracting when we hit end markers

    # Pre-scan headers to see if Introduction exists
    headers = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    intro_exists = any('introduction' in h.get_text(
        strip=True).lower() for h in headers)

    # If no Introduction exists, start from the beginning of main content
    state = BEFORE_INTRO if intro_exists else IN_MAIN_TEXT

    # Section markers that indicate end of main article content
    END_MARKERS = [
        'methods',
        'materials and methods',
        'references',
        'acknowledgements',
        'acknowledgments',
        'additional information',
        'supplementary information',
        'supplementary data',
        'supplementary materials',
        'data availability',
        'author contributions',
        'competing interests',
        'ethics declarations',
        'rights and permissions',
        'about this article',
        'footnotes'
    ]

    # Iterate through all elements in main content
    for element in main_content.descendants:
        # Check if this is a header element
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            header_text = element.get_text(strip=True).lower()

            # Look for Introduction header to start extraction
            if state == BEFORE_INTRO and 'introduction' in header_text:
                state = IN_MAIN_TEXT
                continue

            # Look for Discussion header
            if state == IN_MAIN_TEXT and 'discussion' in header_text:
                state = IN_DISCUSSION
                continue

            # Check for end markers once we're in or past Discussion
            if state in [IN_DISCUSSION, AFTER_DISCUSSION]:
                for marker in END_MARKERS:
                    if marker in header_text:
                        # Stop extraction when we hit an end marker
                        state = AFTER_DISCUSSION
                        break

            # Skip headers themselves
            continue

        # Only process content after Introduction/Results/Abstract is found and before end markers
        if state in [BEFORE_INTRO, AFTER_DISCUSSION]:
            continue

        # Process paragraph-like elements
        if element.name in ['p', 'div', 'section']:
            # Skip if inside figures/tables
            if element.find_parent(['figure', 'figcaption', 'table']):
                continue

            # Skip elements that contain other paragraph containers
            has_paragraph_child = False
            for child in element.find_all(['p', 'div', 'section'], recursive=False):
                has_paragraph_child = True
                break

            if has_paragraph_child:
                continue

            # Extract text
            text = element.get_text(' ', strip=True)
            if text:
                paragraphs.append(text)

    # Join all paragraphs with double newlines
    return '\n\n'.join(paragraphs)


def extract_from_file(html_file_path, min_words=20, max_words=500, verbose=True):
    """
    Extract text paragraphs from a local HTML file.

    This is the main function for programmatic use. It reads a locally saved
    HTML file and extracts clean text paragraphs.

    Args:
        html_file_path: Path to the HTML file to process
        min_words: Minimum words per paragraph (default: 20)
        max_words: Maximum words per paragraph (default: 500)
        verbose: Print progress messages to stderr (default: True)

    Returns:
        tuple: (paragraphs_list, metadata_dict)
        - paragraphs_list: List of paragraph strings (empty if error)
        - metadata_dict: {
            'file_path': str,
            'success': bool,
            'error': str or None,
            'num_paragraphs': int,
            'total_words': int
        }

    Exceptions are caught internally. Check metadata['success'] and 
    metadata['error'] for status.
    """
    metadata = {
        'file_path': html_file_path,
        'success': False,
        'error': None,
        'num_paragraphs': 0,
        'total_words': 0
    }

    try:
        if verbose:
            print(f"Reading HTML file: {html_file_path}", file=sys.stderr)

        # Check if file exists
        if not os.path.exists(html_file_path):
            metadata['error'] = f"File not found: {html_file_path}"
            if verbose:
                print(f"Error: {metadata['error']}", file=sys.stderr)
            return [], metadata

        # Read HTML content from file
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        if verbose:
            print(
                f"File loaded: {len(html_content)} characters", file=sys.stderr)

        # Check if content looks like PDF (safety check)
        content_start = html_content[:1024].lower()
        if '%pdf' in content_start:
            metadata['error'] = "PDF content detected. Only HTML files are supported."
            if verbose:
                print(f"Error: {metadata['error']}", file=sys.stderr)
            return [], metadata

        # Extract paragraphs
        paragraphs = extract_paragraphs_from_html(
            html_content,
            upper_limit=max_words,
            lower_limit=min_words
        )

        if not paragraphs:
            metadata['error'] = "No paragraphs extracted from the content"
            if verbose:
                print(f"Warning: {metadata['error']}", file=sys.stderr)
            return [], metadata

        # Calculate metadata
        metadata['num_paragraphs'] = len(paragraphs)
        metadata['total_words'] = sum(count_words(p) for p in paragraphs)
        metadata['success'] = True

        if verbose:
            print(f"Extracted {len(paragraphs)} paragraph(s)", file=sys.stderr)

        return paragraphs, metadata

    except ValueError as e:
        metadata['error'] = str(e)
        if verbose:
            print(f"Error: {e}", file=sys.stderr)
        return [], metadata

    except UnicodeDecodeError as e:
        metadata['error'] = f"Could not read file (encoding issue): {e}"
        if verbose:
            print(f"Error: {metadata['error']}", file=sys.stderr)
        return [], metadata

    except Exception as e:
        metadata['error'] = f"Unexpected error: {e}"
        if verbose:
            print(f"Unexpected Error: {e}", file=sys.stderr)
        return [], metadata


def extract_full_text_from_html_file(html_file_path, verbose=True):
    """
    Extract complete main text from a local HTML file.

    This function reads an HTML file and returns the full text content
    without filtering short paragraphs or chunking long ones.

    Args:
        html_file_path: Path to the HTML file to process
        verbose: Print progress messages to stderr (default: True)

    Returns:
        tuple: (full_text_string, metadata_dict)
        - full_text_string: Complete text (empty string if error)
        - metadata_dict: {
            'file_path': str,
            'success': bool,
            'error': str or None,
            'num_paragraphs': int,
            'total_words': int
        }
    """
    metadata = {
        'file_path': html_file_path,
        'success': False,
        'error': None,
        'num_paragraphs': 0,
        'total_words': 0
    }

    try:
        if verbose:
            print(f"Reading HTML file: {html_file_path}", file=sys.stderr)

        # Check if file exists
        if not os.path.exists(html_file_path):
            metadata['error'] = f"File not found: {html_file_path}"
            if verbose:
                print(f"Error: {metadata['error']}", file=sys.stderr)
            return "", metadata

        # Read HTML content from file
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        if verbose:
            print(
                f"File loaded: {len(html_content)} characters", file=sys.stderr)

        # Extract full text
        full_text = extract_full_text_from_html(html_content)

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
            print(
                f"Extracted {metadata['num_paragraphs']} paragraph(s), {metadata['total_words']} words", file=sys.stderr)

        return full_text, metadata

    except ValueError as e:
        metadata['error'] = str(e)
        if verbose:
            print(f"Error: {e}", file=sys.stderr)
        return "", metadata

    except UnicodeDecodeError as e:
        metadata['error'] = f"Could not read file (encoding issue): {e}"
        if verbose:
            print(f"Error: {metadata['error']}", file=sys.stderr)
        return "", metadata

    except Exception as e:
        metadata['error'] = f"Unexpected error: {e}"
        if verbose:
            print(f"Unexpected Error: {e}", file=sys.stderr)
        return "", metadata


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Extract clean text paragraphs from an HTML file.'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to the HTML file to process'
    )
    parser.add_argument(
        '--min-words',
        type=int,
        default=20,
        help='Minimum words per paragraph (default: 20)'
    )
    parser.add_argument(
        '--max-words',
        type=int,
        default=500,
        help='Maximum words per paragraph (default: 500)'
    )

    args = parser.parse_args()

    # Use the main function
    paragraphs, metadata = extract_from_file(
        html_file_path=args.input,
        min_words=args.min_words,
        max_words=args.max_words,
        verbose=True
    )

    # Check success
    if not metadata['success']:
        sys.exit(1)

    # Print paragraphs to stdout with formatting
    for i, para in enumerate(paragraphs, 1):
        print(f"\n{'='*60}")
        print(f"Paragraph {i} ({count_words(para)} words)")
        print(f"{'='*60}")
        print(para)

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(
        f"Summary: {metadata['num_paragraphs']} paragraphs, {metadata['total_words']} total words", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    return paragraphs


if __name__ == '__main__':
    main()
