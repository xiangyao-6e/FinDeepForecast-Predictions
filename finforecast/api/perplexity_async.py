#!/usr/bin/env python3
"""
Async Perplexity API Script with Sonar Models

This script provides an async interface to Perplexity's API via OpenRouter with:
- Sonar models for reasoning and search
- Sonar Deep Research for comprehensive analysis
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


class AsyncPerplexityAPI(BaseAPI):
    """Async main class for Perplexity API operations via OpenRouter"""
    
    def __init__(self, function: str):
        """
        Initialize the async Perplexity API client
        
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
        
        # Set model based on function type
        if function == "deep_research":
            self.model = "perplexity/sonar-deep-research"
        
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
                        response_json = await response.json()
                
                # Extract the content from the response
                if 'choices' not in response_json or len(response_json['choices']) == 0:
                    raise KeyError("No choices in response")
                
                final_output = response_json['choices'][0]['message']['content']
                
                # Extract citations if available
                citations = []
                # Perplexity may include citations in different formats
                if response_json.get('citations'):
                    citations = [Citation(
                                title="",
                                url=url) for url in response_json['citations']]
                
                # Check for citations in message metadata
                message_metadata = response_json['choices'][0].get('message', {})
                if 'citations' in message_metadata:
                    citations = [Citation(
                                title=citation.get('title', ''),
                                url=citation.get('url', '')) 
                                for citation in message_metadata['citations']]

                duration = asyncio.get_event_loop().time() - start
                result = DeepResearchResult(
                    prompt=prompt,
                    report=final_output,
                    citations=citations,
                    reasoning_steps=[],
                    web_searches=[],
                    raw_response=response_json,
                    time=duration
                )
                # result.pretty_print(logger)
                logger.info(f"Function {self.function} process completed successfully")
                return result
                
            except KeyError as e:
                logger.error(f"KeyError in {self.function} process (attempt {attempt + 1}): {str(e)}")
                logger.error(f"Response: {response_json if 'response_json' in locals() else 'No response'}")
                if attempt == max_retries - 1:
                    # Last attempt failed, raise the exception
                    logger.error(f"All {max_retries} attempts failed for function {self.function} request")
                    raise Exception(f"API Error: {str(e)}. Response may indicate insufficient balance or invalid API key.")
                else:
                    # Wait before retrying (exponential backoff)
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                    
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

    async def deep_research(self, prompt: str, stream: bool = False, **kwargs):
        """
        Perform deep research using Perplexity Sonar Deep Research model
        
        Args:
            prompt: The research question or topic
            stream: Whether to stream the response (not implemented)
            
        Returns:
            Comprehensive research report as a DeepResearchResult
        """
        payload = {
            "model": "perplexity/sonar-deep-research",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            'reasoning': {
                'effort': kwargs.get('reasoning_effort', 'high')
            },
        }
        
        max_retries = kwargs.get('max_retries', 3)
        
        return await self.handle_result_with_retries(prompt, payload, max_retries)
    
    async def __call__(self, prompt: str, stream: bool = False, **kwargs):
        """
        Call the appropriate function based on initialization
        
        Args:
            prompt: The question or problem to process
            stream: Whether to stream the response
            **kwargs: Additional keyword arguments
            
        Returns:
            DeepResearchResult with the response
        """
        if self.function == "thinking":
            return await self.thinking(prompt, stream, **kwargs)
        elif self.function == "thinking_and_search":
            return await self.thinking_and_search(prompt, stream, **kwargs)
        elif self.function == "deep_research":
            return await self.deep_research(prompt, stream, **kwargs)

