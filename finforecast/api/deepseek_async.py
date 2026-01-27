#!/usr/bin/env python3
"""
Async DeepSeek API Script with DeepSeek V3.2 Model, Reasoning, and Search Functions

This script provides an async interface to DeepSeek's API via OpenRouter with:
- DeepSeek V3.2 model integration
- Reasoning function for step-by-step thinking
- Search function via web plugin (Exa engine)
- Error handling and logging
- Configurable parameters
- Async/await patterns for non-blocking operations
"""

import asyncio
import aiohttp

from env import OPENROUTER_API_KEY

from finforecast.log_util import get_logger
from finforecast.objs import Citation, DeepResearchResult
from finforecast.api.base_api import BaseAPI

from pdb import set_trace

logger = get_logger()

API_KEY = OPENROUTER_API_KEY

class AsyncDeepSeekAPI(BaseAPI):
    """Async main class for DeepSeek API operations with reasoning and search capabilities"""
    
    def __init__(self, function: str):
        """
        Initialize the async DeepSeek API script
        
        Args:
            function: The function type to use ("thinking", "thinking_and_search", "deep_research")
        """

        assert function in ["thinking", "thinking_and_search", "deep_research"]
        self.function = function
        self.api_key = API_KEY

        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        self.model = "deepseek/deepseek-v3.2-exp"
        self.max_tokens = "64000"
    
    async def handle_result_with_retries(self, prompt: str, payload, max_retries: int):
        """Handle API request with retries and exponential backoff"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} of {max_retries} for model {self.model} function {self.function} request")
                start = asyncio.get_event_loop().time()
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.url, 
                        headers=self.headers, 
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=3600)
                    ) as response:
                        
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"API request failed with status {response.status}: {error_text}")
                        
                        result_data = await response.json()
                final_output = result_data['choices'][0]['message']['content']
                
                citations = []

                possible_citations = result_data['choices'][0]['message'].get("annotations")
                if possible_citations:
                    citations = [Citation(
                                title=ann.get('url_citation', {}).get('title', ""),
                                url=ann.get('url_citation', {}).get('url', "")) for ann in possible_citations]

                duration = asyncio.get_event_loop().time() - start
                result = DeepResearchResult(
                    prompt=prompt,
                    report=final_output,
                    citations=citations,
                    reasoning_steps=[],
                    web_searches=[],
                    raw_response=result_data,
                    time=duration
                )
                # result.pretty_print(logger)
                logger.info(f"Function {self.function} process completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {self.function} process (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    # Last attempt failed, raise the exception
                    logger.error(f"All {max_retries} attempts failed for function {self.function} request")
                    raise e
                else:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

    async def thinking(self, prompt: str, stream: bool, **kwargs):
        """
        Perform step-by-step thinking and reasoning using DeepSeek
        
        Args:
            prompt: The problem or question to think about
            stream: Whether to stream the response (not implemented for DeepSeek)
            **kwargs: Additional arguments including max_retries
            
        Returns:
            DeepResearchResult with reasoning output
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "reasoning": {
                "enabled": True,
                "effort": "high"
            },
            "max_tokens": self.max_tokens
        }
        max_retries = kwargs.get('max_retries', 3)
        
        return await self.handle_result_with_retries(prompt, payload, max_retries)
    
    async def thinking_and_search(self, prompt: str, stream: bool, **kwargs):
        """
        Perform thinking with web search capabilities using DeepSeek
        
        Args:
            prompt: The problem or question to think about
            stream: Whether to stream the response (not implemented for DeepSeek)
            **kwargs: Additional arguments including max_retries
            
        Returns:
            DeepResearchResult with reasoning and search output
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "reasoning": {
                "enabled": True,
                "effort": "high"
            },
            "plugins": [
                {
                    "id": "web",
                    "engine": "exa",
                    "max_results": 5
                }
            ],
            "max_tokens": self.max_tokens
        }
        max_retries = kwargs.get('max_retries', 3)
        
        return await self.handle_result_with_retries(prompt, payload, max_retries)

    async def deep_research(self, prompt: str, stream: bool, **kwargs):
        """
        Perform deep research (currently same as thinking_and_search)
        
        Args:
            prompt: The problem or question to research
            stream: Whether to stream the response (not implemented for DeepSeek)
            **kwargs: Additional arguments including max_retries
            
        Returns:
            DeepResearchResult with research output
        """
        # For now, deep research is the same as thinking_and_search
        return await self.thinking_and_search(prompt, stream, **kwargs)
    
    async def __call__(self, prompt: str, stream: bool = False, **kwargs):
        """
        Call the appropriate method based on the function type
        
        Args:
            prompt: The input prompt
            stream: Whether to stream the response
            **kwargs: Additional arguments
            
        Returns:
            DeepResearchResult from the appropriate method
        """
        if self.function == "thinking":
            return await self.thinking(prompt, stream, **kwargs)
        elif self.function == "thinking_and_search":
            return await self.thinking_and_search(prompt, stream, **kwargs)
        elif self.function == "deep_research":
            return await self.deep_research(prompt, stream, **kwargs)

