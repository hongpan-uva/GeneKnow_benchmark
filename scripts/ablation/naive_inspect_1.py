#!/usr/bin/env python3
"""Naive inspection mode pipeline v1 - direct full-text summarization without passage retrieval."""

from geneknow.utils.helpers import (
    validate_project_name, initialize_output_files, resolve_gene_source,
    get_target_paper, write_report_entry, write_usage_entry, early_cancel,
    prepare_gene_context
)
from geneknow.processing.text import extract_paragraphs_from_xml
from geneknow.processing.retrieval import download_paper_xml
from geneknow.api.europepmc import format_apa_from_europepmc
from geneknow.llm.tracker import GPTUsageTracker
from geneknow.core.config import CONFIG
import asyncio
import argparse
import logging
from pathlib import Path
import os
import sys
from openai import AsyncOpenAI
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


async def direct_summarize_fulltext(full_text: str, gene: str, cell_type: str,
                                    tracker: GPTUsageTracker, client,
                                    tracker_lock, semaphore, model="gpt-5-mini"):
    """Directly summarize full paper text focusing on gene-celltype relationship.

    This combines the logic of summarize_passage and summarize_article into a single call.
    """
    async with semaphore:
        response = await client.responses.create(
            model=model,
            input=[{"role": "system",
                    "content": "You are a concise academic assistant. You summarize ONLY from the provided context. Do not add facts or world knowledge. Use plain English, avoid speculation."},
                   {"role": "developer",
                    "content": f"""
Instructions:
- Write a ≤150-word summary of the relationship between {gene} and {cell_type} using only the provided text.
- Prioritize high-level statements and the most specific role first (e.g., specifies, required for, drives identity).
- First determine the relationship structure in the passage (causal chain, intermediary mechanism, parallel listing, or association), then restate it faithfully without merging steps, strengthening claims, or changing hierarchy. Choose connectors based on structure (not fluency); if unsure, use neutral wording.
- Preserve key context and conditions explicitly stated (e.g., developmental stage, disease/injury state, species, tissue/region, experimental setting). Do not add, assume, or generalize beyond the text.
Content Rules:
- First determine whether the passage contains {gene}-centered narrative (systematically elaborates {gene} function in {cell_type}) or non–{gene}-centered (mentions {gene} only peripherally or in independent facts).
• If non–{gene}-centered: summarize only direct statements that explicitly mentions both {gene} and {cell_type}, excluding indirect, background, or contextual information.
• If {gene}-centered: write a summary following a logical flow: role(context) → mechanism → outcome, preserving exact relationship structure and drivers without merging steps.
- Inforamtion to ignore: facts not directly related to {gene}, Assay/protocol details, long gene lists, Over-precise numbers
Output:
- Write in natural prose, NO title, NO subtitles, NO bullet points, No arrows.
- If the passage is methods/figure legend/references, or no {gene}–{cell_type} relationship is present, output: no function revealed
                    """},
                {"role": "user",
                    "content": [
                        {"type": "input_text", "text": f"Summarize the relationship between {gene} and {cell_type} from the following full paper text."},
                        {"type": "input_text", "text": full_text}
                    ]
                 }]
        )

    async with tracker_lock:
        tracker.update(response.usage)

    return response.output_text


async def async_naive_inspect_1(args, locks_dict):
    """Run naive inspection mode v1 asynchronously (direct full-text summarization).

    Args:
        args: Parsed command line arguments
        locks_dict: Dictionary of async locks
    """
    # Set global locks from dict
    global out_report_lock, out_usage_lock, europepmc_lock, ncbi_lock, tracker_lock, error_genes_lock
    out_report_lock = locks_dict['out_report_lock']
    out_usage_lock = locks_dict['out_usage_lock']
    pmc_lock = locks_dict['pmc_lock']
    europepmc_lock = locks_dict['europepmc_lock']
    ncbi_lock = locks_dict['ncbi_lock']
    tracker_lock = locks_dict['tracker_lock']
    error_genes_lock = locks_dict['error_genes_lock']

    print("naive_inspect_1 mode:", args)

    # Validate project name
    if not validate_project_name(args.name):
        logger.error(
            "Project name must contain only alphanumeric characters, hyphens, and underscores.")
        return 1

    # Create project subdirectory
    project_dir = args.outdir / args.name

    ct_all_names = args.context

    # optional parameters
    model_used = CONFIG.DEFAULT_MODEL

    # Initialize output files and directories
    await initialize_output_files(project_dir, "naive_inspect_1")

    # read in gene list
    g_list = resolve_gene_source(args)

    # Load API key from environment variable
    client_async = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"))
    openai_semaphore = asyncio.Semaphore(CONFIG.OPENAI_SEMAPHORE_LIMIT)

    # Initialize error genes tracking
    error_genes = []

    # first-step retrieval: retrieve target paper
    # test PMID: 41271638 PMCID: PMC12639169
    paper_metadata = await get_target_paper(args, europepmc_lock)
    target_pmid = paper_metadata[0]["pmid"]
    target_pmcid = paper_metadata[0]["pmcid"]

    # read full text
    full_text_return = await download_paper_xml(target_pmcid, pmc_lock, europepmc_lock)
    if full_text_return["note"] != "":
        print(full_text_return["note"])
        return 1
    xml_text = full_text_return["xml_text"]

    # build citation
    try:
        citation = format_apa_from_europepmc(paper_metadata)
    except Exception:
        citation = f"PMID: {target_pmid}"

    # Convert XML to full text (no paragraph filtering)
    print("Extracting full text from XML...")
    paragraphs = extract_paragraphs_from_xml(xml_text)
    full_text = "\n\n".join(paragraphs)
    print(
        f"Extracted {len(paragraphs)} paragraphs, total text length: {len(full_text)} characters")

    async def per_gene_inspect(g_primary_name, args, client, semaphore, error_genes):
        report_dict = {"Gene": g_primary_name, "Summary": "",
                       "PMID": target_pmid, "Reference": citation, "Note": ""}
        tracker = GPTUsageTracker()

        # Prepare gene context (aliases and prompt parameters)
        gene_context = await prepare_gene_context(
            g_primary_name, args.species, args.auto_alias,
            ct_all_names, ncbi_lock, client
        )
        g_secondary_names = gene_context["secondary_names"]
        prompt_paramters = gene_context["prompt_parameters"]

        print(
            f"[PMID {target_pmid}] Generating direct summary for {g_primary_name} from full text...")
        try:
            direct_summary = await direct_summarize_fulltext(
                full_text=full_text,
                gene=prompt_paramters["g_insert"],
                cell_type=prompt_paramters["ct_insert"],
                tracker=tracker,
                client=client,
                tracker_lock=tracker_lock,
                semaphore=semaphore,
                model=model_used
            )
        except Exception:
            await early_cancel(project_dir, g_primary_name,
                               "An error occurred while generating the summary.",
                               report_dict, tracker, out_report_lock, out_usage_lock, "naive_inspect_1",
                               error_genes, error_genes_lock)
            return None

        if direct_summary == "" or direct_summary == CONFIG.NO_FUNCTION_MARKER:
            await early_cancel(project_dir, g_primary_name,
                               "No function was revealed by this paper.",
                               report_dict, tracker, out_report_lock, out_usage_lock, "naive_inspect_1",
                               error_genes, error_genes_lock)
            return None

        # Create evidence passages DataFrame with full text and summary
        passage_df = pd.DataFrame({
            "text": [full_text],
            "jr_score": [""],
            "summary": [direct_summary]
        })
        passage_df.to_csv(project_dir / CONFIG.EVIDENCE_PASSAGES_SUBDIR /
                          f"{g_primary_name}_PMID{target_pmid}_evidence_passages.csv", index=False)

        # Skip verification step - use direct_summary directly
        print(
            f"[PMID {target_pmid}] Skipping verification for {g_primary_name} (naive_inspect_1 mode)...")

        report_dict["Summary"] = direct_summary
        # Use helper functions for output
        await write_report_entry(project_dir, report_dict, out_report_lock, "naive_inspect_1")
        await write_usage_entry(project_dir, tracker, g_primary_name, out_usage_lock)

        # Check if report has an error note (only for genes that weren't early cancelled)
        if report_dict.get("Note", "").startswith("An error"):
            async with error_genes_lock:
                error_genes.append(g_primary_name)

        return None

    await asyncio.gather(*(per_gene_inspect(g, args, client_async, openai_semaphore, error_genes) for g in g_list), return_exceptions=True)

    # Write error genes to file
    if error_genes:
        error_genes_path = project_dir / "error_genes.txt"
        with open(error_genes_path, "w") as file:
            file.write("\n".join(error_genes))
        print(f"Error genes written to: {error_genes_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    from geneknow.core.config import CONFIG

    parser = argparse.ArgumentParser(
        prog="naive_inspect_1",
        description="Naive inspection mode v1 - direct full-text summarization without passage retrieval"
    )

    parser.add_argument(
        "-g", "--genes",
        nargs="+",          # one or more values
        help="Space-separated gene symbols (e.g., -g FOXA1 HOXB13). Overrides -G if both are provided."
    )
    parser.add_argument(
        "-G", "--genes-file", type=Path,
        help="Path to a file with one gene symbol per line. Ignored if -g/--genes is provided."
    )
    parser.add_argument(
        "-c", "--context",
        nargs="+",          # one or more values
        required=True,
        help='Required. Space-separated context aliases (e.g., -c "prostate cancer" PCa).'
    )
    parser.add_argument(
        "-o", "--outdir",
        type=Path,
        default=Path("output"),
        help='Output directory. Defaults to output/.'
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        required=True,
        help='Required. Project name (alphanumeric, hyphens, underscores only). Output goes to outdir/name/'
    )
    parser.add_argument(
        "-s", "--species",
        type=str,
        default="human",
        help='Species for gene-alias lookup. Default: human.'
    )
    parser.add_argument(
        "--auto-alias",
        action="store_true",
        dest="auto_alias",
        help='Enable automatic gene alias matching via NCBI Gene.'
    )
    parser.add_argument(
        "--pmid",
        type=str,
        help='PubMed ID of the target paper.'
    )
    parser.add_argument(
        "--pmcid",
        type=str,
        help='PubMed Central ID of the target paper.'
    )
    parser.add_argument(
        "-N", "--suffix-not-allowed",
        action="store_false",
        dest="suffix_allowed",
        help='Disable suffix matching on context terms (e.g., plurals). Only exact context terms will be used. This does not affect gene-name suffix handling.'
    )
    parser.set_defaults(auto_alias=False, suffix_allowed=True)

    return parser


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Setup locks
    out_report_lock = asyncio.Lock()
    out_usage_lock = asyncio.Lock()
    pmc_lock = asyncio.Lock()
    europepmc_lock = asyncio.Lock()
    ncbi_lock = asyncio.Lock()
    tracker_lock = asyncio.Lock()
    error_genes_lock = asyncio.Lock()

    locks_dict = {
        'out_report_lock': out_report_lock,
        'out_usage_lock': out_usage_lock,
        'pmc_lock': pmc_lock,
        'europepmc_lock': europepmc_lock,
        'ncbi_lock': ncbi_lock,
        'tracker_lock': tracker_lock,
        'error_genes_lock': error_genes_lock
    }

    result = asyncio.run(async_naive_inspect_1(args, locks_dict))
    sys.exit(result)


if __name__ == "__main__":
    main()
