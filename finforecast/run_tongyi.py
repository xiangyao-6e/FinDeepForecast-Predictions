#!/usr/bin/env python3
"""
Run AsyncPipelineFinForecast with OpenAI model and deep_research function
"""

import asyncio
import os
import sys
from pathlib import Path
from pdb import set_trace
import json

# Add parent directory to path for imports

from pipeline_async_finforecast import AsyncPipelineFinForecast

async def main():
    """Run OpenAI with deep_research function"""
    stream = False
    forecast_excel_file = Path(__file__).parent.parent / "finforecast_wk10_12_28.xlsx"
    
    model = "tongyi"
    function_name = "deep_research"
    output_dir = Path(__file__).parent.parent / "data/finforecast_wk10_test/deep_research/"
    
    print(f"Starting pipeline with model={model}, function={function_name}")
    print(f"Forecast file: {forecast_excel_file}")
    print(f"Output directory: {output_dir}")

    parent_dir = Path(__file__).resolve().parent.parent
    # Load JSON from parent directory
    sheet_mapping_path = parent_dir / 'sheet_mapping.json'

    sheet_mapping = json.load(open(sheet_mapping_path, 'r'))
    
    pipeline = AsyncPipelineFinForecast(
        model, function_name, stream, str(forecast_excel_file), str(output_dir), sheet_mapping
    )
    # Get all task IDs
    task_ids = list(pipeline.task_metadata.keys())
    print(f"Total tasks: {len(task_ids)}")
    
    # Process with controlled concurrency
    # Adjust max_concurrent based on your needs and API limits
    # Note: Deep research may take longer, so we use lower concurrency
    results = await pipeline.process_tasks_with_semaphore(task_ids, max_concurrent=1)
    
    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {sum(1 for r in results if not isinstance(r, Exception))}")
    print(f"Failed: {sum(1 for r in results if isinstance(r, Exception))}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(main())

