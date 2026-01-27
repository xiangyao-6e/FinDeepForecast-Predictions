#!/usr/bin/env python3
"""
Master script to run all FinForecast tasks

This script can run all 10 configurations or specific subsets based on command-line arguments.
"""

import asyncio
from datetime import datetime

# import sys
# sys.path.insert(0, str(Path(__file__).parent.parent))

from finforecast.log_util import get_logger
from pipeline_async_finforecast import AsyncPipelineFinForecast

logger = get_logger()

class FinForecastRunner:
    def __init__(self, forecast_excel_file: str, output_dir: str, sheet_mapping: str):
        self.models = ["openai", "claude", "grok", "gemini", "gemini3", "deepseek", "perplexity"]
        self.functions = ["thinking", "thinking_and_search", "deep_research"]
        self.concurrency_config = {
            "openai": {"thinking": 10, "thinking_and_search": 10, "deep_research": 10},
            "claude": {"thinking": 3, "thinking_and_search": 3},
            "grok": {"thinking": 10, "thinking_and_search": 10},
            "gemini": {"thinking": 10, "thinking_and_search": 10},
            "gemini3": {"thinking": 10, "thinking_and_search": 10},
            "deepseek": {"thinking": 10, "thinking_and_search": 10},
            "perplexity": {"deep_research": 10},
            "tongyi": {"deep_research": 10}
        }

        # self.concurrency_config = {
        #     "openai": {"thinking": 1, "thinking_and_search": 1, "deep_research": 1},
        #     "claude": {"thinking": 1, "thinking_and_search": 1},
        #     "grok": {"thinking": 1, "thinking_and_search": 1},
        #     "gemini": {"thinking": 1, "thinking_and_search": 1},
        #     "gemini3": {"thinking": 1, "thinking_and_search": 1},
        #     "deepseek": {"thinking": 1, "thinking_and_search": 1},
        #     "perplexity": {"deep_research": 1},
        #     "tongyi": {"deep_research": 1}
        # }

        self.model2stream = {
            "claude": True,
            "openai": False,
            "gemini": False,
            "gemini3": False,
            "grok": True,
            "deepseek": False,
            "perplexity": False,
            "tongyi": False
        }

        self.configurations = [
            (model, function) 
            for model, func_configs in self.concurrency_config.items()
            for function in func_configs.keys()
        ]
        

        self.forecast_excel_file = forecast_excel_file
        self.output_dir = output_dir
        self.sheet_mapping = sheet_mapping
    
    async def run_configuration(self, model: str, function_name: str):
        """Run a single model-function configuration"""
        logger.info(f"task_runner: Starting {model} - {function_name}")
        print(f"\n{'='*80}")
        print(f"Starting: {model} - {function_name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Validate model and function configuration
        if model not in self.model2stream:
            error_msg = f"Unknown model: {model}. Available models: {list(self.model2stream.keys())}"
            raise Exception(f"ERROR: {error_msg}")
            
        if model not in self.concurrency_config:
            error_msg = f"No concurrency config for model: {model}"
            raise Exception(f"ERROR: {error_msg}")
        
        if function_name not in self.concurrency_config[model]:
            error_msg = f"Function '{function_name}' not configured for model '{model}'. Available: {list(self.concurrency_config[model].keys())}"
            raise Exception(f"ERROR: {error_msg}")
        
        stream = self.model2stream[model]
        
        try:
            pipeline = AsyncPipelineFinForecast(
                model, function_name, stream, str(self.forecast_excel_file), str(self.output_dir), self.sheet_mapping
            )
            
            # Get all task IDs
            task_ids = list(pipeline.task_metadata.keys())
            print(f"Total tasks: {len(task_ids)}")
            
            # Get concurrency setting for this configuration
            max_concurrent = self.concurrency_config[model][function_name]
            print(f"Max concurrent: {max_concurrent}\n")
            
            # Process with controlled concurrency
            results = await pipeline.process_tasks_with_semaphore(task_ids, max_concurrent=max_concurrent)
            
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = sum(1 for r in results if isinstance(r, Exception))

            if failed > 0:
                for r in results:
                    if isinstance(r, Exception):
                        print(f"{model}, {function_name}: Error: {r}")
            
            logger.info(f"task_runner: Completed {model} - {function_name}: {len(results)} total, {successful} successful, {failed} failed")
            print(f"\n{'='*80}")
            print(f"Completed: {model} - {function_name}")
            print(f"Total: {len(results)} | Successful: {successful} | Failed: {failed}")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")
            
            return {
                "model": model,
                "function": function_name,
                "total": len(results),
                "successful": successful,
                "failed": failed,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"task_runner: ERROR in {model} - {function_name}: {e}")
            print(f"\n{'='*80}")
            print(f"ERROR in {model} - {function_name}: {str(e)}")
            print(f"{'='*80}\n")
            
            return {
                "model": model,
                "function": function_name,
                "total": 0,
                "successful": 0,
                "failed": 0,
                "status": "error",
                "error": str(e)
            }
    
    async def run_task_generation(self):
        """
        Run FinForecast task generation for all configurations.
        
        Args:
            configurations: List of (model, function) tuples
            parallel: Whether to run configurations in parallel
        
        Returns:
            List of result dictionaries
        """
        logger.info(f"task_runner: GENERATING MARKDOWN FILES - {len(self.configurations)} configurations")
        print(f"\n{'#'*80}")
        print(f"# GENERATING MARKDOWN FILES")
        print(f"#")
        print(f"# Total configurations: {len(self.configurations)}")
        print(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}\n")
        
        start_time = datetime.now()
        
        # Run configurations
        # Run in parallel (all at once)
        tasks = [self.run_configuration(model, function) for model, function in self.configurations]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"task_runner: GENERATION COMPLETE - duration {duration}, {len(results)} configs")
        # Print summary
        print(f"\n{'#'*80}")
        print(f"# GENERATION COMPLETE")
        print(f"#")
        print(f"# Total configurations: {len(results)}")
        print(f"# Duration: {duration}")
        print(f"# End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}\n")
        
        # Print detailed results table
        print(f"{'Model':<12} {'Function':<25} {'Total':<8} {'Success':<10} {'Failed':<8} {'Status':<10}")
        print(f"{'-'*80}")
        
        for result in results:
            if isinstance(result, Exception):
                print(f"{'ERROR':<12} {'N/A':<25} {'0':<8} {'0':<10} {'0':<8} {'exception':<10}")
            else:
                print(
                    f"{result['model']:<12} "
                    f"{result['function']:<25} "
                    f"{result['total']:<8} "
                    f"{result['successful']:<10} "
                    f"{result['failed']:<8} "
                    f"{result['status']:<10}"
                )
        
        print(f"{'-'*80}\n")
        
        return results

# MODELS = ["openai", "claude", "grok", "gemini", "gemini3", "deepseek", "perplexity"]
# FUNCTIONS = ["thinking", "thinking_and_search", "deep_research"]

# # Concurrency settings for each model and function
# CONCURRENCY_CONFIG = {
#     "openai": {"thinking": 10, "thinking_and_search": 10, "deep_research": 10},
#     "claude": {"thinking": 3, "thinking_and_search": 3},
#     "grok": {"thinking": 10, "thinking_and_search": 10},
#     "gemini": {"thinking": 10, "thinking_and_search": 10},
#     "gemini3": {"thinking": 10, "thinking_and_search": 10},
#     "deepseek": {"thinking": 10, "thinking_and_search": 10},
#     "perplexity": {"deep_research": 10},
# }

# model2stream = {
#     "claude": True,
#     "openai": False,
#     "gemini": False,
#     "gemini3": False,
#     "grok": True,
#     "deepseek": False,
#     "perplexity": False,
# }



# async def main():
#     """Main function to orchestrate all runs"""
#     parser = argparse.ArgumentParser(
#         description="Run FinForecast tasks for multiple models and functions"
#     )
#     parser.add_argument(
#         "--models",
#         nargs="+",
#         choices=MODELS,
#         default=MODELS,
#         help="Models to run (default: all)"
#     )
#     parser.add_argument(
#         "--function",
#         choices=FUNCTIONS + ["all"],
#         default="all",
#         help="Function to run (default: all)"
#     )
#     parser.add_argument(
#         "--parallel",
#         action="store_true",
#         help="Run configurations parallel instead of in sequentially"
#     )
    
#     args = parser.parse_args()
    
#     # Determine which functions to run
#     functions = FUNCTIONS if args.function == "all" else [args.function]
    
#     # Create list of configurations to run
#     # configurations = [
#     #     (model, function) 
#     #     for model in args.models 
#     #     for function in functions
#     # ]
#     configurations = [(model, function) for model, values in CONCURRENCY_CONFIG.items() for function in values.keys()]
    
#     print(f"\n{'#'*80}")
#     print(f"# FinForecast Task Batch Runner")
#     print(f"#")
#     print(f"# Models: {', '.join(args.models)}")
#     print(f"# Functions: {', '.join(functions)}")
#     print(f"# Total configurations: {len(configurations)}")
#     print(f"# Mode: {'Sequential' if not args.parallel else 'Parallel'}")
#     print(f"# Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"{'#'*80}\n")
    
#     start_time = datetime.now()
    
#     # Run configurations
#     if args.parallel:
#         # Run in parallel (all at once)
#         tasks = [run_configuration(model, function) for model, function in configurations]
#         results = await asyncio.gather(*tasks, return_exceptions=True)
#     else:
#         # Run sequentially
#         results = []
#         for model, function in configurations:
#             result = await run_configuration(model, function)
#             results.append(result)
    
#     end_time = datetime.now()
#     duration = end_time - start_time
    
#     # Print summary
#     print(f"\n{'#'*80}")
#     print(f"# FINAL SUMMARY")
#     print(f"#")
#     print(f"# Total configurations: {len(results)}")
#     print(f"# Duration: {duration}")
#     print(f"# End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"{'#'*80}\n")
    
#     # Print detailed results table
#     print(f"{'Model':<12} {'Function':<25} {'Total':<8} {'Success':<10} {'Failed':<8} {'Status':<10}")
#     print(f"{'-'*80}")
    
#     total_tasks = 0
#     total_success = 0
#     total_failed = 0
    
#     for result in results:
#         if isinstance(result, Exception):
#             print(f"{'ERROR':<12} {'N/A':<25} {'0':<8} {'0':<10} {'0':<8} {'exception':<10}")
#         else:
#             print(
#                 f"{result['model']:<12} "
#                 f"{result['function']:<25} "
#                 f"{result['total']:<8} "
#                 f"{result['successful']:<10} "
#                 f"{result['failed']:<8} "
#                 f"{result['status']:<10}"
#             )
#             total_tasks += result['total']
#             total_success += result['successful']
#             total_failed += result['failed']
#             # if total_failed > 0:
#             #     from pdb import set_trace;set_trace()
    
#     print(f"{'-'*80}")
#     print(f"{'TOTAL':<12} {'':<25} {total_tasks:<8} {total_success:<10} {total_failed:<8}")
#     print(f"\n{'#'*80}\n")


# if __name__ == "__main__":
#     asyncio.run(main())

