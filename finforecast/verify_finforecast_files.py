#!/usr/bin/env python3
"""
Script to verify file counts in the finforecast directory structure.

This script checks that each model under the three modes (deep_research, thinking, 
thinking_and_search) has the correct number of files in their Answer, markdown, 
and misc directories. The expected counts and task IDs are loaded from an Excel file.

Features:
- Reads task IDs and expected counts from Excel files (e.g., finforecast_wk10.xlsx)
- Supports flexible column naming (task_id, Task ID, TaskID, ID, etc.)
- Can auto-detect Excel files in common locations
- Validates both file counts and specific task IDs
- Shows detailed reports of missing or extra task IDs
- Supports relative and absolute paths
"""

import os
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Pipeline logger (configured by log_util in run_all_with_extraction)
logger = logging.getLogger('pipeline')


# Expected folders under each model
EXPECTED_FOLDERS = ['Answer', 'markdown', 'misc']

# Expected modes
EXPECTED_MODES = ['deep_research', 'thinking', 'thinking_and_search']


def normalize_category_name(name):
    """
    Normalize category names from Excel sheet format to directory format.
    Converts "Corporate - Recurrent" to "Corporate_Recurrent", etc.
    
    Args:
        name (str): Category name from Excel sheet
        
    Returns:
        str: Normalized category name for directory matching
    """
    # Robust normalization to match folder naming:
    # - Trim whitespace
    # - Replace any dash variants surrounded by optional spaces with underscore
    # - Replace remaining spaces with underscore
    # - Collapse multiple underscores
    # - Strip leading/trailing underscores
    # This ensures "Macro - Non Recurrent" → "Macro_Non_Recurrent"
    import re
    if name is None:
        return ""
    normalized = str(name).strip()
    # Replace en/em dashes and hyphens with a single hyphen
    normalized = normalized.replace("—", "-").replace("–", "-")
    # Replace any occurrences of spaces around hyphen with single underscore
    normalized = re.sub(r"\s*-\s*", "_", normalized)
    # Replace remaining whitespace groups with underscore
    normalized = re.sub(r"\s+", "_", normalized)
    # Replace any remaining non-alphanumeric (except underscore) with underscore
    normalized = re.sub(r"[^\w]", "_", normalized)
    # Collapse multiple underscores
    normalized = re.sub(r"_+", "_", normalized)
    # Trim underscores
    normalized = normalized.strip("_")
    return normalized


def load_expected_data_from_excel(excel_path):
    """
    Load expected counts and task IDs from the Excel file.
    
    Args:
        excel_path (Path): Path to the Excel file
        
    Returns:
        dict: Dictionary with expected counts and task ID sets for each category
    """
    expected_data = {}
    
    try:
        xls = pd.ExcelFile(excel_path)
        
        logger.info(f"verify_finforecast_files: load_expected_data reading {excel_path}, {len(xls.sheet_names)} sheet(s)")
        print(f"Reading Excel file: {excel_path}")
        print(f"Found {len(xls.sheet_names)} sheet(s): {', '.join(xls.sheet_names)}")
        
        for sheet_name in xls.sheet_names:
            # Read the sheet
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Remove trailing spaces from sheet name
            clean_name = sheet_name.strip()
            
            # Normalize the name to match directory structure
            normalized_name = normalize_category_name(clean_name)
            
            # Try to find task ID column with different possible names
            task_id_col = None
            possible_names = ['task_id', 'Task ID', 'TaskID', 'ID', 'id', 'Task_ID', 'task id']
            
            for col_name in possible_names:
                if col_name in df.columns:
                    task_id_col = col_name
                    break
            
            if task_id_col:
                # Filter out NaN values and convert to int
                task_ids = df[task_id_col].dropna().astype(int).tolist()
                task_ids = set(task_ids)
                
                expected_data[normalized_name] = {
                    'count': len(task_ids),
                    'task_ids': task_ids,
                    'original_name': clean_name
                }
                logger.info(f"verify_finforecast_files: load_expected_data sheet '{clean_name}' → '{normalized_name}': {len(task_ids)} task IDs")
                print(f"  Sheet '{clean_name}' → '{normalized_name}': {len(task_ids)} task IDs")
            else:
                logger.warning(f"verify_finforecast_files: No task ID column in sheet '{sheet_name}'")
                print(f"  Warning: No task ID column found in sheet '{sheet_name}'")
                print(f"    Available columns: {', '.join(df.columns.tolist())}")
                
    except Exception as e:
        logger.exception(f"verify_finforecast_files: load_expected_data error: {e}")
        print(f"Error reading Excel file: {e}")
        raise
    
    # Summary of loaded expected data
    total_task_ids = sum(data['count'] for data in expected_data.values())
    logger.info(f"verify_finforecast_files: load_expected_data done - {len(expected_data)} categories, {total_task_ids} total task IDs")
    return expected_data


def count_files_in_directory(directory):
    """
    Count the number of files (not directories) in a directory.
    
    Args:
        directory (Path): Path to the directory
        
    Returns:
        int: Number of files in the directory
    """
    if not directory.exists():
        return 0
    
    try:
        return len([f for f in directory.iterdir() if f.is_file()])
    except Exception as e:
        print(f"    Error reading {directory}: {e}")
        return 0


def get_task_ids_in_directory(directory):
    """
    Get the set of task IDs (file names without extensions) in a directory.
    
    Args:
        directory (Path): Path to the directory
        
    Returns:
        set: Set of task IDs (as integers)
    """
    if not directory.exists():
        return set()
    
    try:
        task_ids = set()
        for f in directory.iterdir():
            if f.is_file():
                # Get the filename without extension
                task_id_str = f.stem
                try:
                    task_ids.add(int(task_id_str))
                except ValueError:
                    # Skip files that don't have numeric names
                    pass
        return task_ids
    except Exception as e:
        print(f"    Error reading {directory}: {e}")
        return set()


def verify_model_directory(model_path, mode_name, expected_data):
    """
    Verify that a model directory has the correct file counts and task IDs.
    
    Args:
        model_path (Path): Path to the model directory
        mode_name (str): Name of the mode (for reporting)
        expected_data (dict): Expected counts and task IDs for each category
        
    Returns:
        dict: Dictionary with verification results
    """
    model_name = model_path.name
    results = {
        'model_name': model_name,
        'mode_name': mode_name,
        'folders': {},
        'has_issues': False
    }
    
    # Check each expected folder (Answer, markdown, misc)
    for folder_name in EXPECTED_FOLDERS:
        folder_path = model_path / folder_name
        results['folders'][folder_name] = {}
        
        if not folder_path.exists():
            results['has_issues'] = True
            results['folders'][folder_name]['exists'] = False
            continue
        
        results['folders'][folder_name]['exists'] = True
        
        # Check each category (Corporate, Macro, etc.)
        for category_name, category_data in expected_data.items():
            subdir_path = folder_path / category_name
            expected_count = category_data['count']
            expected_task_ids = category_data['task_ids']
            
            if not subdir_path.exists():
                results['folders'][folder_name][category_name] = {
                    'exists': False,
                    'count': 0,
                    'expected': expected_count,
                    'match': False,
                    'missing_ids': expected_task_ids,
                    'extra_ids': set()
                }
                results['has_issues'] = True
            else:
                actual_count = count_files_in_directory(subdir_path)
                actual_task_ids = get_task_ids_in_directory(subdir_path)
                
                # Check if counts match
                count_match = (actual_count == expected_count)
                
                # Check if task IDs match
                missing_ids = expected_task_ids - actual_task_ids
                extra_ids = actual_task_ids - expected_task_ids
                ids_match = (len(missing_ids) == 0 and len(extra_ids) == 0)
                
                match = count_match and ids_match
                
                results['folders'][folder_name][category_name] = {
                    'exists': True,
                    'count': actual_count,
                    'expected': expected_count,
                    'match': match,
                    'missing_ids': missing_ids,
                    'extra_ids': extra_ids
                }
                
                if not match:
                    results['has_issues'] = True
    
    return results


def verify_mode_directory(mode_path, expected_data):
    """
    Verify all model directories within a mode directory.
    
    Args:
        mode_path (Path): Path to the mode directory
        expected_data (dict): Expected counts and task IDs for each category
        
    Returns:
        list: List of verification results for each model
    """
    mode_name = mode_path.name
    results = []
    
    if not mode_path.exists():
        logger.info(f"verify_finforecast_files: verify_mode_directory '{mode_name}' - path does not exist")
        return results
    
    # Get all subdirectories (model directories)
    model_dirs = sorted([d for d in mode_path.iterdir() if d.is_dir()])
    logger.info(f"verify_finforecast_files: verify_mode_directory '{mode_name}' - {len(model_dirs)} model(s)")
    
    for model_dir in model_dirs:
        result = verify_model_directory(model_dir, mode_name, expected_data)
        results.append(result)
    
    issues = sum(1 for r in results if r['has_issues'])
    if issues:
        logger.info(f"verify_finforecast_files: verify_mode_directory '{mode_name}' - {issues}/{len(results)} model(s) with issues")
    return results


def print_results(all_results, expected_data, verbose=False, show_ids=False):
    """
    Print verification results in a readable format.
    
    Args:
        all_results (list): List of all verification results
        expected_data (dict): Expected counts and task IDs for each category
        verbose (bool): If True, show all models. If False, only show models with issues
        show_ids (bool): If True, show missing and extra task IDs
    """
    print("="*80)
    print("FINFORECAST FILE COUNT VERIFICATION")
    print("="*80)
    
    # Print expected counts (using original names for readability)
    expected_summary = ", ".join([
        f"{data.get('original_name', cat)}={data['count']}" 
        for cat, data in expected_data.items()
    ])
    logger.info(f"verify_finforecast_files: verification started - expected counts: {expected_summary}")
    print(f"Expected counts: {expected_summary}")
    print()
    
    # Group results by mode
    results_by_mode = defaultdict(list)
    for result in all_results:
        results_by_mode[result['mode_name']].append(result)
    
    total_models = 0
    models_with_issues = 0
    
    for mode_name in EXPECTED_MODES:
        if mode_name not in results_by_mode:
            logger.info(f"verify_finforecast_files: mode {mode_name} - directory not found")
            print(f"\n{'='*80}")
            print(f"Mode: {mode_name.upper()}")
            print(f"{'='*80}")
            print("  ⚠ Mode directory not found")
            continue
        
        mode_results = results_by_mode[mode_name]
        mode_issues = sum(1 for r in mode_results if r['has_issues'])
        
        logger.info(f"verify_finforecast_files: mode {mode_name} - {len(mode_results)} model(s), {mode_issues} with issues")
        print(f"\n{'='*80}")
        print(f"Mode: {mode_name.upper()}")
        print(f"{'='*80}")
        print(f"Models: {len(mode_results)} | Issues: {mode_issues}")
        
        total_models += len(mode_results)
        models_with_issues += mode_issues
        
        for result in mode_results:
            # Skip models without issues if not in verbose mode
            if not verbose and not result['has_issues']:
                continue
            
            model_name = result['model_name']
            
            if result['has_issues']:
                # Build breakdown of issues for logging
                issue_parts = []
                for folder_name in EXPECTED_FOLDERS:
                    folder_data = result['folders'].get(folder_name, {})
                    if not folder_data.get('exists', False):
                        issue_parts.append(f"{folder_name}/:MISSING")
                        continue
                    for category_name in expected_data.keys():
                        subdir_data = folder_data.get(category_name, {})
                        if not subdir_data.get('exists', False):
                            issue_parts.append(f"{folder_name}/{category_name}:MISSING")
                        elif not subdir_data.get('match', False):
                            c, e = subdir_data.get('count', 0), subdir_data.get('expected', 0)
                            issue_parts.append(f"{folder_name}/{category_name}:{c} vs expected {e}")
                logger.info(f"verify_finforecast_files: mode {mode_name} model {model_name} - issues: {'; '.join(issue_parts)}")
                print(f"\n  ❌ {model_name}")
            else:
                if verbose:
                    logger.info(f"verify_finforecast_files: mode {mode_name} model {model_name} - OK")
                print(f"\n  ✓ {model_name}")
            
            # Print details for each folder
            for folder_name in EXPECTED_FOLDERS:
                folder_data = result['folders'].get(folder_name, {})
                
                if not folder_data.get('exists', False):
                    # from pdb import set_trace
                    # set_trace()
                    print(f"      {folder_name}/: MISSING")
                    continue
                
                # Check all categories
                for category_name in expected_data.keys():
                    subdir_data = folder_data.get(category_name, {})
                    
                    if not subdir_data.get('exists', False):
                        # from pdb import set_trace
                        # set_trace()
                    
                        print(f"      {folder_name}/{category_name}/: MISSING")
                    elif not subdir_data.get('match', False):
                        count = subdir_data.get('count', 0)
                        expected = subdir_data.get('expected', 0)
                        diff = count - expected
                        sign = '+' if diff > 0 else ''
                        print(f"      {folder_name}/{category_name}/: {count} files (expected {expected}, {sign}{diff})")
                        
                        # Show missing and extra IDs if requested
                        if show_ids:
                            missing_ids = subdir_data.get('missing_ids', set())
                            extra_ids = subdir_data.get('extra_ids', set())
                            
                            if missing_ids:
                                missing_str = ', '.join(map(str, sorted(missing_ids)[:10]))
                                if len(missing_ids) > 10:
                                    missing_str += f'... ({len(missing_ids)} total)'
                                print(f"        Missing IDs: {missing_str}")
                            
                            if extra_ids:
                                extra_str = ', '.join(map(str, sorted(extra_ids)[:10]))
                                if len(extra_ids) > 10:
                                    extra_str += f'... ({len(extra_ids)} total)'
                                print(f"        Extra IDs: {extra_str}")
                    elif verbose:
                        count = subdir_data.get('count', 0)
                        print(f"      {folder_name}/{category_name}/: {count} files ✓")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total models checked: {total_models}")
    print(f"Models with correct counts: {total_models - models_with_issues}")
    print(f"Models with issues: {models_with_issues}")
    
    logger.info(f"verify_finforecast_files: SUMMARY - total_models={total_models}, correct={total_models - models_with_issues}, with_issues={models_with_issues}")
    if models_with_issues == 0:
        logger.info(f"verify_finforecast_files: SUMMARY - all models correct")
        print("\n✓ All models have the correct file counts and task IDs!")
    else:
        logger.info(f"verify_finforecast_files: SUMMARY - {models_with_issues} model(s) have issues")
        print(f"\n⚠ {models_with_issues} model(s) have incorrect file counts or task IDs")
    
    print("="*80)


def main():
    """Main function to verify file counts across all modes and models."""
    parser = argparse.ArgumentParser(
        description='Verify file counts and task IDs in finforecast directory structure',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all models (show only those with issues)
  python verify_finforecast_files.py
  
  # Show all models, including those without issues
  python verify_finforecast_files.py --verbose
  
  # Show missing and extra task IDs
  python verify_finforecast_files.py --show-ids
  
  # Use a custom base directory
  python verify_finforecast_files.py --base-dir /path/to/finforecast_wk10
  
  # Use a custom Excel file
  python verify_finforecast_files.py --excel finforecast_wk10.xlsx
  
  # Auto-detect Excel file
  python verify_finforecast_files.py --excel auto
        """
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        default='/Users/hxy/workspace/finddr_finforecast/data/baseline/finforecast_wk10',
        help='Base directory containing finforecast data (default: /Users/hxy/workspace/finddr_finforecast/data/baseline/finforecast_wk10)'
    )
    
    parser.add_argument(
        '--excel',
        type=str,
        default='/Users/hxy/workspace/finddr_finforecast/finforecast_wk10.xlsx',
        help='Path to Excel file with expected task IDs (default: /Users/hxy/workspace/finddr_finforecast/finforecast_wk10.xlsx). Use "auto" to auto-detect.'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show all models, including those without issues'
    )
    
    parser.add_argument(
        '--show-ids',
        action='store_true',
        help='Show missing and extra task IDs for models with issues'
    )
    
    args = parser.parse_args()
    
    finforecast_dir = Path(args.base_dir).expanduser().resolve()
    
    # Handle Excel file path
    # if args.excel.lower() == 'auto':
    #     # excel_path = find_excel_file(finforecast_dir)
    #     if excel_path is None:
    #         print("Error: Could not auto-detect Excel file")
    #         print("Tried looking in:")
    #         print(f"  - {finforecast_dir}/finforecast_wk10.xlsx")
    #         print(f"  - {finforecast_dir}/finforecast.xlsx")
    #         print(f"  - {finforecast_dir}/forecast.xlsx")
    #         print(f"  - {finforecast_dir.parent}/finforecast_wk10.xlsx")
    #         print(f"  - {Path.cwd()}/finforecast_wk10.xlsx")
    #         return 1
    #     print(f"Auto-detected Excel file: {excel_path}\n")
    # else:
    excel_path = Path(args.excel).expanduser().resolve()
    
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
    print_results(all_results, expected_data, verbose=args.verbose, show_ids=args.show_ids)
    
    # Return exit code based on whether issues were found
    has_issues = any(r['has_issues'] for r in all_results)
    return 1 if has_issues else 0


if __name__ == "__main__":
    exit(main())

