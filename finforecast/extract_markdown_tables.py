#!/usr/bin/env python3
"""
Script to extract values from markdown tables.

This script reads markdown files from all models under the finforecast directory's
three modes (deep_research, thinking, thinking_and_search), identifies the last table
with 2 columns and 2 rows (header + 1 data row), and extracts the value from the 
2nd column of the 2nd row (the data row). The extracted values are written to text 
files in the 'Answer' directory structure.
"""

import re
import json
import logging

# Pipeline logger (configured by log_util in run_all_with_extraction)
logger = logging.getLogger('pipeline')


class MarkdownTableExtractor:
    # Subdirectories to process in markdown and answer directories
    SUBDIRS = [
        'Corporate', 'Macro', 
        'Corporate_Recurrent', 'Corporate_Non_Recurrent',
        'Macro_Recurrent', 'Macro_Contingent', 'Macro_Non_Recurrent'
    ]
    
    @staticmethod
    def parse_markdown_tables(content):
        """
        Parse markdown tables from content and return a list of parsed tables.
        
        Each table is represented as a list of rows, where each row is a list of cell values.
        
        Args:
            content (str): The markdown content to parse
            
        Returns:
            list: List of tables, where each table is a list of rows
        """
        tables = []
        lines = content.split('\n')
        current_table = []
        in_table = False
        
        for line in lines:
            # Check if line looks like a table row (contains pipes)
            if '|' in line:
                # Parse the row
                cells = [cell.strip() for cell in line.split('|')]
                # Remove empty cells from start and end (markdown tables often have leading/trailing |)
                cells = [cell for cell in cells if cell]
                
                # Skip separator lines (e.g., |-------|-------|)
                if cells and all(re.match(r'^[\s\-:]+$', cell) for cell in cells):
                    continue
                
                if cells:
                    current_table.append(cells)
                    in_table = True
            else:
                # End of table
                if in_table and current_table:
                    tables.append(current_table)
                    current_table = []
                    in_table = False
        
        # Don't forget the last table if file ends with a table
        if current_table:
            tables.append(current_table)
        
        return tables


    @staticmethod
    def find_last_2col_2row_table(tables):
        """
        Find the last table that has exactly 2 columns and 2 rows (header + 1 data row).
        
        Args:
            tables (list): List of parsed tables
            
        Returns:
            list or None: The last matching table, or None if not found
        """
        for table in reversed(tables):
            # Check if table has exactly 2 rows
            if len(table) == 2:
                # Check if all rows have exactly 2 columns
                if all(len(row) == 2 for row in table):
                    return table
        return None

    @staticmethod
    def extract_value_from_table(table):
        """
        Extract the value from the 2nd column, 2nd row of the table.
        
        Args:
            table (list): A parsed table (list of rows)
            
        Returns:
            str: The extracted value
        """
        def extract_first_numeric(string):
            """Extract only the first numeric value from string"""
            string = string.strip()
            
            # Match the first numeric value (with optional $, -, commas, decimal, %, x)
            pattern = r'^[-$]?[0-9,]+\.?[0-9]*%?x?'
            
            match = re.match(pattern, string)
            return match.group(0) if match else None

        def is_valid_binary(string):
            """Check if string is YES or NO (case insensitive)"""
            return string.strip().upper() in ['YES', 'NO']
        
        if table and len(table) >= 2 and len(table[1]) >= 2:
            value = table[1][1]
            if is_valid_binary(value):
                return value.strip().upper()  # Return normalized YES/NO
            # Default numeric validation for other sheets
            if extract_first_numeric(value):
                return value
        return None

    @classmethod
    def process_markdown_file(cls, input_path, output_path, force_output=False):
        """
        Process a single markdown file and extract the table value.
        
        Args:
            input_path (Path): Path to the input markdown file
            output_path (Path): Path to the output text file
            force_output (bool): If True, create empty Answer file even when no value can be extracted
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read the markdown file
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse tables
            tables = cls.parse_markdown_tables(content)
            
            # Find the last table with 2 columns and 2 rows
            target_table = cls.find_last_2col_2row_table(tables)
            
            if target_table is None:
                print(f"  Warning: No suitable table found in {input_path.name}")
                if force_output:
                    # Create empty Answer file
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write('')
                    return True
                return False
            
            # Extract the value
            value = cls.extract_value_from_table(target_table)
            
            if value is None:
                print(f"  Warning: Could not extract value from {input_path.name}")
                if force_output:
                    # Create empty Answer file
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write('')
                    return True
                return False
            
            # Write to output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(value)
            
            return True
            
        except Exception as e:
            print(f"  Error processing {input_path.name}: {e}")
            return False

    @classmethod
    def process_model_directory(cls, model_path, force_output=False):
        """
        Process all markdown files in a model directory.
        
        Args:
            model_path (Path): Path to the model directory
            force_output (bool): If True, create empty Answer file even when no value can be extracted
            
        Returns:
            tuple: (total_processed, total_success, sheet_stats)
                    sheet_stats is a dict with keys as sheet names and values as (processed, success) tuples
        """
        model_name = model_path.name
        markdown_dir = model_path / "markdown"
        answer_dir = model_path / "Answer"
        
        if not markdown_dir.exists():
            return 0, 0, {}
        
        print(f"\n  Processing model: {model_name}")
        print(f"    Input:  {markdown_dir}")
        print(f"    Output: {answer_dir}")
        
        total_processed = 0
        total_success = 0
        sheet_stats = {}
        
        # Process all subdirectories
        for subdir in MarkdownTableExtractor.SUBDIRS:
            input_subdir = markdown_dir / subdir
            output_subdir = answer_dir / subdir
            
            if not input_subdir.exists():
                continue
            
            # Get all markdown files
            md_files = sorted(input_subdir.glob('*.md'))
            
            if md_files:
                print(f"    Processing {subdir}: {len(md_files)} files...")
            
            sheet_processed = 0
            sheet_success = 0
            
            for md_file in md_files:
                # Create output filename (same name but .txt extension)
                output_file = output_subdir / (md_file.stem + '.txt')
                
                # Process the file
                sheet_processed += 1
                total_processed += 1
                if cls.process_markdown_file(md_file, output_file, force_output=force_output):
                    sheet_success += 1
                    total_success += 1
            
            if sheet_processed > 0:
                sheet_stats[subdir] = (sheet_processed, sheet_success)
        
        if total_processed > 0:
            logger.info(f"extract_markdown_tables: model {model_name}: {total_success}/{total_processed} files OK" + (
                f", sheet breakdown: {dict((k, f'{s}/{p}') for k, (p, s) in sheet_stats.items())}" if sheet_stats else ""
            ))
        print(f"    ✓ {model_name}: {total_success}/{total_processed} files processed successfully")
        
        return total_processed, total_success, sheet_stats

    @classmethod
    def process_mode_directory(cls, mode_path, model_failures=None, sheet_failures=None, force_output=False):
        """
        Process all model directories within a mode directory.
        
        Args:
            mode_path (Path): Path to the mode directory
            model_failures (dict, optional): Dictionary to track failures per model and mode
            sheet_failures (dict, optional): Dictionary to track failures per sheet
            force_output (bool): If True, create empty Answer file even when no value can be extracted
            
        Returns:
            tuple: (total_processed, total_success)
        """
        mode_name = mode_path.name
        logger.info(f"extract_markdown_tables: mode {mode_name} - starting")
        print(f"\n{'='*70}")
        print(f"Processing mode: {mode_name.upper()}")
        print(f"{'='*70}")
        
        total_processed = 0
        total_success = 0
        
        # Get all subdirectories (model directories)
        model_dirs = [d for d in mode_path.iterdir() if d.is_dir()]
        
        if not model_dirs:
            logger.info(f"extract_markdown_tables: mode {mode_name} - no model directories")
            print(f"  No model directories found in {mode_name}")
            return 0, 0
        
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name
            processed, success, sheet_stats = cls.process_model_directory(model_dir, force_output=force_output)
            total_processed += processed
            total_success += success
            
            # Track failures per model and mode if tracking dictionary is provided
            if model_failures is not None:
                fails = processed - success
                if fails > 0:
                    key = f"{model_name}_{mode_name}"
                    model_failures[key] = fails
            
            # Aggregate sheet statistics
            if sheet_failures is not None:
                for sheet_name, (sheet_proc, sheet_succ) in sheet_stats.items():
                    sheet_fails = sheet_proc - sheet_succ
                    if sheet_name not in sheet_failures:
                        sheet_failures[sheet_name] = 0
                    sheet_failures[sheet_name] += sheet_fails
        
        logger.info(f"extract_markdown_tables: mode {mode_name} - done: {total_success}/{total_processed} files")
        return total_processed, total_success

    @staticmethod
    def find_failed_markdown_files(model_path):
        """
        Find markdown files that don't have corresponding Answer files.
        Also finds their corresponding misc files.
        
        Args:
            model_path (Path): Path to the model directory
            
        Returns:
            list: List of tuples (md_file_path, misc_file_path) for files without Answer files
                misc_file_path will be None if it doesn't exist
        """

        def check_misc_file_exists(misc_file):
            is_exists = misc_file.exists()
            is_json_loadable = False
            try:
                with open(misc_file, 'r', encoding='utf-8') as f:
                    json.load(f)
                is_json_loadable = True
            except Exception as e:
                pass
            return is_exists and is_json_loadable
        
        def check_answer_file_exists(answer_file):
            return answer_file.exists()
        
        markdown_dir = model_path / "markdown"
        answer_dir = model_path / "Answer"
        misc_dir = model_path / "misc"
        
        if not markdown_dir.exists():
            return []
        
        failed_files = []
        
        for subdir in MarkdownTableExtractor.SUBDIRS:
            md_subdir = markdown_dir / subdir
            answer_subdir = answer_dir / subdir
            misc_subdir = misc_dir / subdir
            
            if not md_subdir.exists():
                continue
            
            # Get all markdown files
            md_files = sorted(md_subdir.glob('*.md'))
            
            for md_file in md_files:
                # Check if corresponding .txt file exists in Answer directory
                answer_file = answer_subdir / (md_file.stem + '.txt')
                misc_file = misc_subdir / (md_file.stem + '.json')
                if not check_misc_file_exists(misc_file) or not check_answer_file_exists(answer_file):
                    failed_files.append((md_file, misc_file))
                    
        return failed_files

    @classmethod
    def cleanup_failed_files(cls, finforecast_dir, dry_run=True):
        """
        Delete markdown files and their corresponding misc files that failed to generate Answer files.
        
        Args:
            finforecast_dir (Path): Base finforecast directory
            dry_run (bool): If True, only show what would be deleted without actually deleting
            
        Returns:
            int: Number of file pairs deleted (or would be deleted in dry-run mode)
        """
        modes = ['deep_research', 'thinking', 'thinking_and_search']
        
        logger.info(f"extract_markdown_tables: cleanup_failed_files started (dry_run={dry_run}), base_dir={finforecast_dir}")
        print("="*70)
        if dry_run:
            print("DRY-RUN MODE: FINDING FILES THAT WOULD BE DELETED")
        else:
            print("CLEANUP MODE: DELETING FAILED MARKDOWN AND MISC FILES")
        print("="*70)
        print(f"Base directory: {finforecast_dir}")
        
        all_failed_files = []
        
        # Process each mode
        for mode_name in modes:
            mode_path = finforecast_dir / mode_name
            
            if not mode_path.exists():
                continue
            
            mode_failed_count = 0
            print(f"\n{'='*70}")
            print(f"Mode: {mode_name.upper()}")
            print(f"{'='*70}")
            
            # Get all model directories
            model_dirs = [d for d in mode_path.iterdir() if d.is_dir()]
            
            for model_dir in sorted(model_dirs):
                model_name = model_dir.name
                failed_files = cls.find_failed_markdown_files(model_dir)
                
                if failed_files:
                    mode_failed_count += len(failed_files)
                    logger.info(f"extract_markdown_tables: cleanup mode {mode_name} model {model_name}: {len(failed_files)} failed file(s)")
                    print(f"\n  Model: {model_name}")
                    print(f"  Found {len(failed_files)} failed file pair(s):")
                    
                    for md_file, misc_file in failed_files:
                        relative_md_path = md_file.relative_to(finforecast_dir)
                        print(f"    - {relative_md_path}")
                        if misc_file:
                            relative_misc_path = misc_file.relative_to(finforecast_dir)
                            print(f"      + {relative_misc_path}")
                        else:
                            print(f"      (no corresponding misc file)")
                        all_failed_files.append((md_file, misc_file))
            
            if mode_failed_count > 0:
                logger.info(f"extract_markdown_tables: cleanup mode {mode_name} total: {mode_failed_count} failed file(s)")
        
        # Summary
        misc_count = sum(1 for _, misc in all_failed_files if misc is not None)
        total_files = len(all_failed_files) + misc_count
        logger.info(f"extract_markdown_tables: cleanup SUMMARY - failed markdown: {len(all_failed_files)}, misc: {misc_count}, total to delete: {total_files}")
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total failed markdown files found: {len(all_failed_files)}")
        print(f"Total corresponding misc files found: {misc_count}")
        print(f"Total files to delete: {total_files}")
        
        if all_failed_files:
            if dry_run:
                logger.info(f"extract_markdown_tables: cleanup dry_run - no files deleted")
                print("\nTo actually delete these files, run with --delete flag")
            else:
                print("\nDeleting files...")
                deleted_count = 0
                
                for md_file, misc_file in all_failed_files:
                    # Delete markdown file
                    try:
                        md_file.unlink()
                        deleted_count += 1
                        print(f"  Deleted: {md_file.name}")
                    except Exception as e:
                        print(f"  Error deleting {md_file}: {e}")
                    
                    # Delete misc file if it exists
                    if misc_file:
                        try:
                            misc_file.unlink()
                            deleted_count += 1
                            print(f"  Deleted: {misc_file.name}")
                        except Exception as e:
                            print(f"  Error deleting {misc_file}: {e}")
                
                logger.info(f"extract_markdown_tables: cleanup deleted {deleted_count}/{total_files} files")
                print(f"\nSuccessfully deleted {deleted_count}/{total_files} files")
        else:
            print("\nNo failed files found. All markdown files have corresponding Answer files!")
        
        logger.info(f"extract_markdown_tables: cleanup_failed_files done - {len(all_failed_files)} failed file(s) found/deleted")
        print("="*70)
        
        return len(all_failed_files)

    @classmethod
    def extract_tables(cls, finforecast_dir, force_output=False):
        """
        Extract tables from all markdown files across all modes and models into Answer directories.
        
        Args:
            finforecast_dir (Path): Base finforecast directory
            force_output (bool): If True, create empty Answer file even when no value can be extracted
        """
        modes = ['deep_research', 'thinking', 'thinking_and_search']
        
        logger.info(f"extract_markdown_tables: extract_tables started, base_dir={finforecast_dir}, force_output={force_output}")
        print("="*70)
        print("MARKDOWN TABLE EXTRACTION SCRIPT")
        print("="*70)
        print(f"Base directory: {finforecast_dir}")
        print(f"Modes to process: {', '.join(modes)}")
        if force_output:
            print("Force output: ENABLED (will create empty Answer files for failed extractions)")
        
        grand_total_processed = 0
        grand_total_success = 0
        model_failures = {}  # Track failures per model and mode combination
        sheet_failures = {}  # Track failures per sheet type
        
        # Process each mode
        for mode_name in modes:
            mode_path = finforecast_dir / mode_name
            
            if not mode_path.exists():
                logger.warning(f"extract_markdown_tables: mode directory does not exist: {mode_path}")
                print(f"\nWarning: Mode directory does not exist: {mode_path}")
                continue
            
            processed, success = cls.process_mode_directory(mode_path, model_failures, sheet_failures, force_output=force_output)
            grand_total_processed += processed
            grand_total_success += success
            logger.info(f"extract_markdown_tables: extract_tables mode {mode_name}: {success}/{processed} files")
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"Total files processed: {grand_total_processed}")
        print(f"Successfully extracted: {grand_total_success}")
        total_failures = grand_total_processed - grand_total_success
        print(f"Failed: {total_failures}")
        
        # Print failures by sheet type
        if sheet_failures:
            print("\n" + "="*70)
            print("FAILURE BREAKDOWN BY SHEET TYPE")
            print("="*70)
            sheet_total_failures = sum(sheet_failures.values())
            for sheet_name in sorted(sheet_failures.keys()):
                count = sheet_failures[sheet_name]
                percentage = (count / sheet_total_failures * 100) if sheet_total_failures > 0 else 0
                logger.info(f"extract_markdown_tables: sheet_failures {sheet_name}: {count} ({percentage:.1f}%)")
                print(f"{sheet_name}: {count} failures ({percentage:.1f}%)")
            logger.info(f"extract_markdown_tables: sheet_failures total: {sheet_total_failures}")
            print(f"\nTotal sheet failures: {sheet_total_failures}")
            if sheet_total_failures != total_failures:
                print(f"Note: Sheet total ({sheet_total_failures}) differs from overall total ({total_failures})")
        
        # Print models with at least 1 fail (by model_mode combination)
        if model_failures:
            print("\n" + "="*70)
            print("MODELS WITH FAILURES")
            print("="*70)
            for model_mode in sorted(model_failures.keys()):
                cnt = model_failures[model_mode]
                logger.info(f"extract_markdown_tables: model_failures {model_mode}: {cnt}")
                print(f"{model_mode}: {cnt} failures")
        
        logger.info(f"extract_markdown_tables: extract_tables done - processed {grand_total_processed}, success {grand_total_success}, failed {grand_total_processed - grand_total_success}")
        print("="*70)


# def main():
#     """Main function to process all markdown files across all modes and models."""
#     parser = argparse.ArgumentParser(
#         description='Extract markdown tables or cleanup failed files',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Extract tables from markdown files (default behavior)
#   python extract_markdown_tables.py
  
#   # Extract tables and create empty Answer files for failed extractions
#   python extract_markdown_tables.py --force-output
  
#   # Preview which Answer directories would be removed (dry-run)
#   python extract_markdown_tables.py --clean-answers
  
#   # Remove all Answer directories before extracting tables
#   python extract_markdown_tables.py --clean-answers --delete
  
#   # Preview which failed markdown files would be deleted (dry-run)
#   python extract_markdown_tables.py --cleanup
  
#   # Actually delete failed markdown files
#   python extract_markdown_tables.py --cleanup --delete
#         """
#     )
    
#     parser.add_argument(
#         '--cleanup',
#         action='store_true',
#         help='Clean up markdown files that failed to generate Answer files'
#     )
    
#     parser.add_argument(
#         '--delete',
#         action='store_true',
#         help='Actually delete files/directories (use with --cleanup or --clean-answers). Without this flag, runs in dry-run mode'
#     )
    
#     parser.add_argument(
#         '--clean-answers',
#         action='store_true',
#         help='Remove all existing Answer directories. Use --delete to actually remove, otherwise runs in dry-run mode'
#     )
    
#     parser.add_argument(
#         '--force-output',
#         action='store_true',
#         help='Generate Answer files even when no valid values can be extracted (creates empty files in those cases)'
#     )
    
#     parser.add_argument(
#         '--base-dir',
#         type=str,
#         default='/Users/hxy/workspace/finddr_finforecast/data/baseline/finforecast_wk10',
#         help='Base directory containing finforecast data (default: /Users/hxy/workspace/finddr_finforecast/data/baseline/finforecast)'
#     )
    
#     args = parser.parse_args()
    
#     finforecast_dir = Path(args.base_dir)
    
#     if not finforecast_dir.exists():
#         print(f"Error: Base directory does not exist: {finforecast_dir}")
#         return
    
#     if args.cleanup:
#         # Cleanup mode
#         dry_run = not args.delete
#         cleanup_failed_files(finforecast_dir, dry_run=dry_run)
#     else:
#         # Extract mode (default)
        
#         # Remove all Answer directories if requested
#         if args.clean_answers:
#             dry_run = not args.delete
#             remove_all_answer_directories(finforecast_dir, dry_run=dry_run)
            
#             # If in dry-run mode, don't proceed with extraction
#             if dry_run:
#                 return
        
#         extract_tables(finforecast_dir, force_output=args.force_output)


# if __name__ == "__main__":
#     main()

