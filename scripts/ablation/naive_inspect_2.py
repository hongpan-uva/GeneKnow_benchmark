#!/usr/bin/env python3
"""Naive inspection mode v2 - passage retrieval without verification."""

import asyncio
import argparse
import logging
from pathlib import Path
import os
import sys
from openai import AsyncOpenAI

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from geneknow.core.config import CONFIG
from geneknow.llm.tracker import GPTUsageTracker
from geneknow.llm.prompts import summarize_passage, summarize_article
from geneknow.api.europepmc import format_apa_from_europepmc
from geneknow.processing.retrieval import get_evidence_passages, download_paper_xml
from geneknow.utils.helpers import (
    validate_project_name, initialize_output_files, resolve_gene_source,
    get_target_paper, write_report_entry, write_usage_entry, early_cancel,
    prepare_gene_context
)

logger = logging.getLogger(__name__)


async def async_naive_inspect_2(args, locks_dict):
    """Run naive inspection mode v2 asynchronously.

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

    print("naive_inspect_2 mode:", args)

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
    await initialize_output_files(project_dir, "inspect")

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

        passage_df = get_evidence_passages(
            xml_full_text=xml_text,
            gene_names=[g_primary_name] + g_secondary_names,
            celltype_names=ct_all_names,
            top_n_keep=args.max_passages,
            celltype_suffix_allowed=args.suffix_allowed
        )

        if passage_df is None or len(passage_df) == 0:
            await early_cancel(project_dir, g_primary_name, "No relevant passages.",
                               report_dict, tracker, out_report_lock, out_usage_lock, "inspect",
                               error_genes, error_genes_lock)
            return None

        print(
            f"[PMID {target_pmid}] Generating passage summaries for {g_primary_name}...")
        passage_summaries = await asyncio.gather(*(summarize_passage(psg, tracker, prompt_paramters, tracker_lock, semaphore, model=model_used) for psg in passage_df["text"]), return_exceptions=True)
        passage_df["summary"] = passage_summaries

        passage_df.to_csv(project_dir / CONFIG.EVIDENCE_PASSAGES_SUBDIR /
                          f"{g_primary_name}_PMID{target_pmid}_evidence_passages.csv", index=False)

        valid_indices = passage_df[passage_df["summary"]
                                   != CONFIG.NO_FUNCTION_MARKER].index
        if len(valid_indices) == 0:
            await early_cancel(project_dir, g_primary_name,
                               "No function was revealed by this paper.",
                               report_dict, tracker, out_report_lock, out_usage_lock, "inspect",
                               error_genes, error_genes_lock)
            return None

        print(
            f"[PMID {target_pmid}] Generating the article summary for {g_primary_name}...")
        try:
            article_summary = await summarize_article(
                passage_df.loc[valid_indices, "summary"], tracker, prompt_paramters, tracker_lock, semaphore, model=model_used)
        except Exception:
            await early_cancel(project_dir, g_primary_name,
                               "An error occurred while generating the summary.",
                               report_dict, tracker, out_report_lock, out_usage_lock, "inspect",
                               error_genes, error_genes_lock)
            return None

        if article_summary == "" or article_summary == CONFIG.NO_FUNCTION_MARKER:
            await early_cancel(project_dir, g_primary_name,
                               "No function was revealed by this paper.",
                               report_dict, tracker, out_report_lock, out_usage_lock, "inspect",
                               error_genes, error_genes_lock)
            return None

        print(
            f"[PMID {target_pmid}] Skipping verification for {g_primary_name} (naive_inspect_2 mode)...")

        # Save final output as unverified article summary
        report_dict["Summary"] = article_summary
        await write_report_entry(project_dir, report_dict, out_report_lock, "inspect")
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
        prog="naive_inspect_2",
        description="Naive inspection mode v2 - passage retrieval without verification"
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
        "--max-passages",
        type=int,
        default=CONFIG.DEFAULT_MAX_PASSAGES,
        help='Max evidence passages to review per paper. Default: 3.'
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

    result = asyncio.run(async_naive_inspect_2(args, locks_dict))
    sys.exit(result)


if __name__ == "__main__":
    main()
