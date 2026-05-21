import argparse
import pathlib
import pymupdf4llm
import sys


def convert_pdf_to_md(input_dir: str):
    """
    Convert all PDF files in input_dir to markdown files.
    Print the list of converted files and their word counts.
    """
    input_path = pathlib.Path(input_dir)

    # Validate directory exists
    if not input_path.exists():
        print(f"Error: Directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not input_path.is_dir():
        print(f"Error: Path '{input_dir}' is not a directory.")
        sys.exit(1)

    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'.")
        return

    # Filter out PDFs that already have corresponding .md files
    pdfs_to_convert = []
    skipped_pdfs = []

    for pdf_file in pdf_files:
        md_file = pdf_file.with_suffix(".md")
        if md_file.exists():
            skipped_pdfs.append(pdf_file)
        else:
            pdfs_to_convert.append(pdf_file)

    print(f"Found {len(pdf_files)} PDF(s) in: {input_path.absolute()}")
    print(f"  - {len(pdfs_to_convert)} to convert")
    print(f"  - {len(skipped_pdfs)} already have .md files (skipped)\n")

    if not pdfs_to_convert:
        print("No new PDFs to convert.")
        return

    results = []
    success_count = 0
    failed_count = 0

    for pdf_file in sorted(pdfs_to_convert):
        try:
            # Convert PDF to markdown
            result = pymupdf4llm.to_markdown(str(pdf_file))

            # Handle different return types (str or list of dicts)
            if isinstance(result, list):
                # Extract text from list of page dictionaries
                md_text = "\n\n".join(
                    page.get("text", "") for page in result
                )
            else:
                md_text = result

            # Generate output path (same name, .md extension)
            md_file = pdf_file.with_suffix(".md")

            # Write markdown file
            md_file.write_text(md_text, encoding="utf-8")

            # Count words
            word_count = len(md_text.split())

            # Store result (filename without .pdf extension)
            filename_stem = pdf_file.stem
            results.append((filename_stem, word_count))

            success_count += 1

        except Exception as e:
            print(f"  Error converting '{pdf_file.name}': {e}")
            failed_count += 1

    # Print results
    if results:
        print("Results (newly converted):")
        print("-" * 50)
        for i, (filename, word_count) in enumerate(results, 1):
            # Format word count with commas
            formatted_count = f"{word_count:,}"
            print(f"{i:3d}. {filename:<30} {formatted_count:>10} words")
        print("-" * 50)

    # Print summary
    print(f"\nSummary: Converted {success_count}, Skipped {len(skipped_pdfs)}, Failed {failed_count} of {len(pdf_files)} PDF(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PDF files to Markdown format."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing PDF files to convert"
    )

    args = parser.parse_args()
    convert_pdf_to_md(args.input_dir)
