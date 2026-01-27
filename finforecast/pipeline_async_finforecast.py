import asyncio
from api.openai_async import AsyncOpenAIAPI
from api.claude_async_openrouter import AsyncClaudeOpenRouterAPI
from api.grok_async import AsyncGrok4API
from api.gemini_async import AsyncGeminiAPI
from api.gemini3_async import AsyncGemini3API
from api.deepseek_async import AsyncDeepSeekAPI
from api.perplexity_async import AsyncPerplexityAPI
from api.tongyi_async import AsyncTongyiAPI

import os
from pathlib import Path
import json
import pandas as pd
from pdb import set_trace

API_MAPPING = {
    "openai": AsyncOpenAIAPI,
    "claude": AsyncClaudeOpenRouterAPI,
    "grok": AsyncGrok4API,
    "gemini": AsyncGeminiAPI,
    "gemini3": AsyncGemini3API,
    "deepseek": AsyncDeepSeekAPI,
    "perplexity": AsyncPerplexityAPI,
    "tongyi": AsyncTongyiAPI
}

class AsyncPipelineFinForecast:
    def __init__(self, model_name: str, function_name: str, stream: bool, forecast_excel_file: str, output_dir: str, sheet_mapping: dict):
        assert function_name in ["thinking", "thinking_and_search", "deep_research"]
        self.api = API_MAPPING[model_name](function_name)
        self.stream = stream
        
        # if sheet_mapping_path is None:
        #     script_dir = os.path.dirname(os.path.abspath(__file__))
        #     # Go up one level to parent directory
        #     parent_dir = os.path.dirname(script_dir)
        #     # Load JSON from parent directory
        #     sheet_mapping_path = os.path.join(parent_dir, 'sheet_mapping.json')
        self.sheet_mapping = sheet_mapping
        
        # Load all 3 sheets from the Excel file
        self.task_metadata = self._load_forecast_data(forecast_excel_file)
        
        self.output_dir = Path(output_dir) / function_name        
        self.model_name = model_name
        
    def _load_forecast_data(self, forecast_excel_file: str):
        """Load task data from forecast.xlsx (Corporate - Recurrent, Macro - Recurrent, Macro - Contingent sheets) and create metadata structure"""
        task_metadata = {}
        sheet_mapping = self.sheet_mapping

        for sheet_name, location in sheet_mapping.items():
            try:
                df = pd.read_excel(forecast_excel_file, sheet_name=sheet_name)
            except:
                print(f"Error loading sheet {sheet_name}")
                continue
            for _, row in df.iterrows():
                task_id = str(row['task_id'])
                task_metadata[task_id] = {
                    'task_question': row['task_question'],
                    'location': location
                }
        
        return task_metadata

    def preprocess(self, task_id):
        """Get the task question for the given task_id"""
        # Simply return the task_question as the prompt
        return self.task_metadata[task_id]['task_question']
    
    async def run(self, prompt: str):
        return await self.api(prompt, self.stream)

    def postprocess(self, response, task_id):
        prompt = response.prompt
        report = response.report
        citations = response.citations
        reasoning_steps = response.reasoning_steps
        web_searches = response.web_searches
        raw_response = response.raw_response
        time = response.time

        task_info = self.task_metadata[task_id]
        location = task_info['location']
        
        # Create output directory based on location (Corporate/Macro)
        task_output_dir = self.output_dir / self.model_name / "markdown" / location
        task_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report with task_id as filename
        with open(task_output_dir / f"{task_id}.md", "w") as f:
            f.write(report)

        # Convert bytes objects to strings for JSON serialization
        def convert_bytes_to_string(obj):
            if isinstance(obj, bytes):
                return obj.decode('utf-8', errors='ignore')
            elif isinstance(obj, dict):
                return {k: convert_bytes_to_string(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_bytes_to_string(item) for item in obj]
            else:
                return obj
        
        metadata_collection = {
            "citations": [str(citation) for citation in citations],
            "reasoning_steps": reasoning_steps,
            "web_searches": web_searches,
            "raw_response": convert_bytes_to_string(raw_response),
            "time": time
        }
        metadata_dir = self.output_dir / self.model_name / "misc" / location
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_dir / f"{task_id}.json", "w") as f:
            json.dump(metadata_collection, f)
        
        return response

    async def __call__(self, task_id: str):
        prompt = self.preprocess(task_id)

        task_info = self.task_metadata[task_id]
        location = task_info['location']
        
        task_output_dir = self.output_dir / self.model_name / "markdown" / location
        
        # Skip if already processed
        if os.path.exists(task_output_dir / f"{task_id}.md"):
            return
        
        response = await self.run(prompt)
        self.postprocess(response, task_id)

    async def process_tasks(self, task_ids: list):
        """Process multiple tasks concurrently"""
        tasks = [self(task_id) for task_id in task_ids]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def process_tasks_with_semaphore(self, task_ids: list, max_concurrent: int = 5):
        """Process multiple tasks with controlled concurrency using semaphore"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(task_id):
            async with semaphore:
                print(f"Processing task_id {task_id} (concurrent limit: {max_concurrent})")
                return await self(task_id)
        
        tasks = [process_with_semaphore(task_id) for task_id in task_ids]
        return await asyncio.gather(*tasks, return_exceptions=True)
