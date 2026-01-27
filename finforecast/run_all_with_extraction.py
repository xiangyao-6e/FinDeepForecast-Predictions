#!/usr/bin/env python3
"""
Orchestration script that combines cleanup, task generation, and table extraction.

This script:
1. Optionally cleans up Answer directories or failed markdown files
2. Runs FinForecast tasks to generate markdown files
3. Extracts values from markdown tables to Answer files

Usage:
    # Full pipeline: cleanup -> generate -> extract
    python run_all_with_extraction.py --clean-answers --delete
    
    # Just generate -> extract (skip cleanup)
    python run_all_with_extraction.py
    
    # Cleanup failed files after extraction
    python run_all_with_extraction.py --cleanup-failed --delete
"""
import os
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

from finforecast.log_util import setup_logger
import json

def verify_finforecast_files(forecast_excel_file, finforecast_dir):
    from verify_finforecast_files import load_expected_data_from_excel, verify_mode_directory, print_results, EXPECTED_MODES
    finforecast_dir = Path(finforecast_dir)
    excel_path = Path(forecast_excel_file)
    
    if not finforecast_dir.exists():
        print(f"Error: Base directory does not exist: {finforecast_dir}")
        return 1
    
    if not excel_path.exists():
        print(f"Error: Excel file does not exist: {excel_path}")
        # # Try auto-detection as fallback
        # auto_excel = find_excel_file(finforecast_dir)
        # if auto_excel:
        #     print(f"However, found Excel file at: {auto_excel}")
        #     print("Use --excel auto or specify the correct path")
        # return 1
        raise Exception(f"Error: Excel file does not exist: {excel_path}")
    
    # Load expected data from Excel file
    try:
        expected_data = load_expected_data_from_excel(excel_path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return 1
    
    if not expected_data:
        print("Error: No data loaded from Excel file")
        return 1
    
    # Collect results from all modes
    all_results = []
    
    for mode_name in EXPECTED_MODES:
        mode_path = finforecast_dir / mode_name
        mode_results = verify_mode_directory(mode_path, expected_data)
        all_results.extend(mode_results)
    
    # Print results
    print_results(all_results, expected_data, verbose=True, show_ids=True)
    
    # Return exit code based on whether issues were found
    has_issues = any(r['has_issues'] for r in all_results)
    return 1 if has_issues else 0

async def main():
    """Main orchestration function"""
    parser = argparse.ArgumentParser(
        description="Orchestrate cleanup, task generation, and table extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: cleanup -> generate -> extract
  python run_all_with_extraction.py --clean-answers --delete
  
  # Just generate -> extract (skip cleanup)
  python run_all_with_extraction.py
  
  # Generate -> extract -> cleanup failed files
  python run_all_with_extraction.py --cleanup-failed --delete
  
  # Custom models and functions
  python run_all_with_extraction.py --models openai claude --function thinking
        """
    )
    # Required arguments
    parser.add_argument(
        "--forecast-excel-file",
        type=str,
        required=True,
        help="Path to the forecast Excel file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--sheet-mapping-path",
        type=str,
        required=False,
        default=None,
        help="Path to the sheet mapping JSON file. If not provided, it will use the default sheet mapping in the parent directory."
    )
    args = parser.parse_args()

    if args.sheet_mapping_path is None:
        parent_dir = Path(__file__).resolve().parent.parent
        # Load JSON from parent directory
        args.sheet_mapping_path = parent_dir / 'sheet_mapping.json'
    
    sheet_mapping = json.load(open(args.sheet_mapping_path, 'r'))
    output_dir_path = Path(args.output_dir)
    
    if not output_dir_path.exists():
        # print(f"Error: Output directory does not exist: {output_dir_path}")
        # print(f"Please create it or specify a different directory with --output-dir")
        # return
        os.makedirs(output_dir_path, exist_ok=True)
    
    # Set up logging
    logger = setup_logger(output_dir_path=output_dir_path)
    print(f"sheet_mapping: {sheet_mapping}")
    logger.info(f"sheet_mapping: {sheet_mapping}")

    # Have to setup logger first before imports
    from extract_markdown_tables import MarkdownTableExtractor
    from task_runner import FinForecastRunner

    runner = FinForecastRunner(forecast_excel_file=args.forecast_excel_file, output_dir=args.output_dir, sheet_mapping=sheet_mapping)

    print(f"\n{'#'*80}")
    print(f"# FinForecast Pipeline Orchestrator")
    print(f"#")
    print(f"# Forecast Excel file: {args.forecast_excel_file}")
    print(f"# Output directory: {args.output_dir}")
    print(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")
    
    logger.info("="*80)
    logger.info("FinForecast Pipeline Orchestrator - Started")
    logger.info(f"Forecast Excel file: {args.forecast_excel_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    overall_start = datetime.now()
    
    # STEP 1: Cleanup Answer directories (optional)
    # if args.clean_answers:
    #     print(f"\n{'#'*80}")
    #     print(f"# STEP 1: CLEANING UP ANSWER DIRECTORIES")
    #     print(f"{'#'*80}\n")
    #     
    #     dry_run = not args.delete
    #     remove_all_answer_directories(finforecast_dir, dry_run=dry_run)
    #     
    #     if dry_run:
    #         print("\nRun with --delete flag to actually remove directories.")
    #         print("Skipping generation and extraction in dry-run mode.")
    #         return
    
    max_retries = 3
    generation_results_all_attempts = []
    cleanup_results_all_attempts = []
    
    for attempt in range(max_retries):
        print(f"\n{'#'*80}")
        print(f"Running task generation (Attempt {attempt + 1}/{max_retries})")
        print(f"{'#'*80}\n")
        
        generation_results = await runner.run_task_generation()
        generation_results_all_attempts.append({
            'attempt': attempt + 1,
            'results': generation_results
        })
        
        # Log generation summary
        total_configs = len(generation_results)
        total_tasks = sum(r.get('total', 0) for r in generation_results if not isinstance(r, Exception))
        total_successful = sum(r.get('successful', 0) for r in generation_results if not isinstance(r, Exception))
        total_failed = sum(r.get('failed', 0) for r in generation_results if not isinstance(r, Exception))
        exceptions = sum(1 for r in generation_results if isinstance(r, Exception))
        
        logger.info(f"GENERATION - Attempt {attempt + 1}/{max_retries}: "
                   f"{total_configs} configs, {total_tasks} tasks, "
                   f"{total_successful} successful, {total_failed} failed, "
                   f"{exceptions} exceptions")

        print(f"\n{'#'*80}")
        print(f"CLEANING UP FAILED MARKDOWN FILES (Attempt {attempt + 1}/{max_retries})")
        print(f"{'#'*80}\n")
        
        num_failed_files = MarkdownTableExtractor.cleanup_failed_files(
            Path(args.output_dir), dry_run=False
        )
        cleanup_results_all_attempts.append({
            'attempt': attempt + 1,
            'num_failed_files': num_failed_files
        })
        
        # Log cleanup summary
        logger.info(f"CLEANUP - Attempt {attempt + 1}/{max_retries}: "
                   f"{num_failed_files} failed files found/deleted")
        
        verify_result = verify_finforecast_files(forecast_excel_file=args.forecast_excel_file, finforecast_dir=args.output_dir)
        if verify_result == 0 and num_failed_files == 0:
            print(f"All files processed and verified successfully after {attempt + 1} attempt(s)")
            logger.info(f"SUCCESS - All files processed and verified after {attempt + 1} attempt(s)")
            break
        elif num_failed_files > 0:
            print(f"Found {num_failed_files} failed files. Retrying...")
            logger.info(f"RETRY - Found {num_failed_files} failed files")
        elif verify_result > 0:
            print(f"Found issues in the finforecast files. Retrying...")
            logger.info(f"RETRY - Found issues in finforecast files (verify_result={verify_result})")
        else:
            print(f"Retrying due to unknown issues...")
            logger.info(f"RETRY - Unknown issues detected")

    logger.info("EXTRACTION - Starting table extraction")
    MarkdownTableExtractor.extract_tables(Path(args.output_dir), force_output=True)
    logger.info("EXTRACTION - Table extraction completed")

    logger.info("VERIFICATION - Completing final verification")
    final_verify_result = verify_finforecast_files(forecast_excel_file=args.forecast_excel_file, finforecast_dir=args.output_dir)
    if final_verify_result == 0:
        print("Final verification successful: All files are correct.")
        logger.info("FINAL VERIFICATION - SUCCESS: All files are correct.")
    else:
        print("Final verification found issues in the files.")
        logger.info("FINAL VERIFICATION - ISSUES FOUND in files.")
    
    # Final summary
    overall_end = datetime.now()
    overall_duration = overall_end - overall_start
    
    print(f"\n{'#'*80}")
    print(f"# PIPELINE COMPLETE")
    print(f"#")
    print(f"# Total duration: {overall_duration}")
    print(f"# End time: {overall_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")
    
    # Log final summary
    logger.info("="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total duration: {overall_duration}")
    logger.info(f"End time: {overall_end.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Generation attempts: {len(generation_results_all_attempts)}")
    logger.info(f"Cleanup attempts: {len(cleanup_results_all_attempts)}")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())

