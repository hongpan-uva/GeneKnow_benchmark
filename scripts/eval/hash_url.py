import hashlib
import base64
import csv
import sys
import subprocess
from pathlib import Path


def url_to_id(url: str, n_bytes: int = 12) -> str:
    """
    Map a URL to a short, URL-safe ID.
    n_bytes controls length: 12 bytes -> 16 chars (base64url without padding).
    """
    h = hashlib.blake2b(url.encode("utf-8"), digest_size=n_bytes).digest()
    return base64.urlsafe_b64encode(h).decode("ascii").rstrip("=")


def process_references(reference_path: str):
    """
    Process all *_references.csv files in the given path.
    
    Args:
        reference_path: Path to directory containing *_references.csv files
    """
    ref_dir = Path(reference_path)
    
    if not ref_dir.exists():
        print(f"Error: Path '{reference_path}' does not exist.")
        sys.exit(1)
    
    if not ref_dir.is_dir():
        print(f"Error: Path '{reference_path}' is not a directory.")
        sys.exit(1)
    
    # Find all *_references.csv files
    csv_files = list(ref_dir.glob("*_references.csv"))
    
    if not csv_files:
        print(f"No *_references.csv files found in '{reference_path}'.")
        return
    
    # Set to track unique (url, hash_id) pairs
    url_hash_pairs = set()
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        
        rows = []
        fieldnames = None
        
        # Read the CSV file
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            
            # Add hash_id column if not present
            if 'hash_id' not in fieldnames:
                fieldnames = list(fieldnames) + ['hash_id']
            
            for row in reader:
                # Check real_reference first - if false, skip hashing
                real_reference_value = row.get('real_reference', '').lower()
                
                if real_reference_value == 'false':
                    # Skip hashing for non-real references
                    row['hash_id'] = ''
                    rows.append(row)
                    continue
                
                # Determine which URL to hash based on real_url field
                real_url_value = row.get('real_url', '').lower()
                
                if real_url_value == 'false':
                    # Use alternative_url
                    url_to_hash = row.get('alternative_url', '')
                else:
                    # Use url (default case, including when real_url is 'true' or empty)
                    url_to_hash = row.get('url', '')
                
                # Generate hash_id
                if url_to_hash:
                    hash_id = url_to_id(url_to_hash)
                else:
                    hash_id = ''
                
                row['hash_id'] = hash_id
                rows.append(row)
                
                # Track unique pairs (using the URL that was hashed)
                if url_to_hash and hash_id:
                    url_hash_pairs.add((url_to_hash, hash_id))
        
        # Write updated CSV back
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"  Updated: {csv_file.name}")
    
    # Generate url_id.csv
    url_id_file = ref_dir / "url_id.csv"
    
    with open(url_id_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['url', 'hash_id'])
        
        # Sort for consistent output
        for url, hash_id in sorted(url_hash_pairs):
            writer.writerow([url, hash_id])
    
    print(f"\nGenerated: {url_id_file}")
    print(f"Total unique URL-hash pairs: {len(url_hash_pairs)}")
    
    # Check for missing HTML files
    papers_dir = ref_dir / "papers"
    if papers_dir.exists() and papers_dir.is_dir():
        missing_files = []

        for url, hash_id in sorted(url_hash_pairs):
            htm_path = papers_dir / f"{hash_id}.htm"
            html_path = papers_dir / f"{hash_id}.html"
            md_path = papers_dir / f"{hash_id}.md"

            if not htm_path.exists() and not html_path.exists() and not md_path.exists():
                missing_files.append((hash_id, url))

        if missing_files:
            print(f"\n=== Missing paper files ({len(missing_files)} hash_ids) ===")
            print(f"\nPress Enter to process each missing file (or 'q' + Enter to quit)...")

            processed_count = 0
            for i, (hash_id, url) in enumerate(missing_files, 1):
                user_input = input(f"\n[{i}/{len(missing_files)}] Press Enter to open: {url}")

                if user_input.lower() == 'q':
                    print("Quitting...")
                    break

                processed_count = i

                # Open URL in default browser using wslview
                try:
                    subprocess.run(['wslview', url], check=True)
                    print(f"  Opened: {url}")
                except subprocess.CalledProcessError as e:
                    print(f"  Error opening URL: {e}")
                except FileNotFoundError:
                    print(f"  Error: wslview not found. Make sure it's installed.")

                # Copy hash_id to clipboard using clip.exe
                try:
                    subprocess.run(['clip.exe'], input=hash_id, text=True, check=True)
                    print(f"  Copied to clipboard: {hash_id}")
                except subprocess.CalledProcessError as e:
                    print(f"  Error copying to clipboard: {e}")
                except FileNotFoundError:
                    print(f"  Error: clip.exe not found. Make sure you're in WSL.")

            print(f"\nProcessed {processed_count} of {len(missing_files)} missing files.")
        else:
            print(f"\nAll {len(url_hash_pairs)} hash_ids have corresponding files (.htm/.html/.md) in papers/")
    else:
        print(f"\nNote: papers/ directory not found at '{papers_dir}'")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python hash_url.py <reference_path>")
        print("Example: python hash_url.py ../discover_cases/case1/references")
        sys.exit(1)
    
    reference_path = sys.argv[1]
    process_references(reference_path)
