#!/usr/bin/env python3
"""
Async Claude API Script with OpenAI-Compatible API Pattern

This script provides an async interface to Claude via OpenRouter API with:
- Claude Sonnet 4.5 model integration
- Thinking function for step-by-step reasoning
- Search function for information retrieval
- OpenAI-compatible API pattern using aiohttp
- Error handling and logging
- Configurable parameters
- Async/await patterns for non-blocking operations
"""

import json
import asyncio
import aiohttp

from env import OPENROUTER_API_KEY

from finforecast.log_util import get_logger
from finforecast.objs import Citation, DeepResearchResult
from finforecast.api.base_api import BaseAPI


from pdb import set_trace

logger = get_logger()

API_KEY = OPENROUTER_API_KEY


class AsyncClaudeOpenRouterAPI(BaseAPI):
    """Async main class for Claude API operations via OpenRouter with OpenAI-compatible API"""
    
    def __init__(self, function: str):
        """
        Initialize the async Claude OpenAI API client
        
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
        
        # Claude Sonnet 4.5 model via OpenRouter
        # self.model = "anthropic/claude-sonnet-4.5-20250929"
        self.model = "anthropic/claude-sonnet-4.5"
        
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
                    # set_trace()
                    raise KeyError("No choices in response")
                
                message = response_json['choices'][0]['message']
                final_output = message['content']
                
                # Extract reasoning steps, web searches, and citations
                reasoning_steps = []
                web_searches = []
                citations = []
                
                # Check for thinking blocks in metadata
                if 'thinking' in message:
                    reasoning_steps.append(message['thinking'])
                
                # Check for tool calls (web searches)
                if 'tool_calls' in message:
                    for tool_call in message['tool_calls']:
                        if tool_call.get('type') == 'web_search' or tool_call.get('function', {}).get('name') == 'web_search':
                            args = tool_call.get('function', {}).get('arguments', '{}')
                            if isinstance(args, str):
                                args = json.loads(args)
                            if 'query' in args:
                                web_searches.append(args['query'])
                
                # Check for citations in metadata
                if 'citations' in message:
                    citations = [Citation(
                                title=citation.get('title', ''),
                                url=citation.get('url', '')) 
                                for citation in message['citations']]
                
                # Check for citations at response level
                if response_json.get('citations'):
                    citations = [Citation(
                                title="",
                                url=url) for url in response_json['citations']]

                duration = asyncio.get_event_loop().time() - start
                result = DeepResearchResult(
                    prompt=prompt,
                    report=final_output,
                    citations=citations,
                    reasoning_steps=reasoning_steps,
                    web_searches=list(set(web_searches)),
                    raw_response=response_json,
                    time=duration
                )
                # result.pretty_print(logger)
                logger.info(f"Function {self.function} process completed successfully")
                logger.info(f"Function {self.function} duration: {duration} seconds")
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

    async def thinking(self, prompt: str, stream: bool = False, **kwargs):
        """
        Perform step-by-step thinking and reasoning
        
        Args:
            prompt: The problem or question to think about
            stream: Whether to stream the response (not implemented)
            **kwargs: Additional keyword arguments
                - budget_tokens: Token budget for thinking (default: 10000)
                - max_tokens: Maximum tokens in response (default: 20000)
                - max_retries: Maximum retry attempts (default: 3)
            
        Returns:
            Detailed thinking process as a DeepResearchResult
        """
        budget_tokens = kwargs.get('budget_tokens', 10000)
        max_tokens = kwargs.get('max_tokens', 20000)
        max_retries = kwargs.get('max_retries', 3)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            # Note: OpenRouter's API may not support thinking parameters in the same way
            # This is a placeholder for potential future support
            # "thinking": {
            #     "type": "enabled",
            #     "budget_tokens": budget_tokens
            # }
            'reasoning': {
                'max_tokens': budget_tokens  # Allocate specific token count for reasoning
                # Alternatively use 'effort': 'high' for approximate allocation
            }
        }
        
        return await self.handle_result_with_retries(prompt, payload, max_retries)

    async def thinking_and_search(self, prompt: str, stream: bool = False, **kwargs):
        """
        Perform thinking with web search capabilities
        
        Args:
            prompt: The problem or question to think about
            stream: Whether to stream the response (not implemented)
            **kwargs: Additional keyword arguments
                - budget_tokens: Token budget for thinking (default: 10000)
                - max_tokens: Maximum tokens in response (default: 20000)
                - max_retries: Maximum retry attempts (default: 3)
            
        Returns:
            Detailed thinking process with search results as a DeepResearchResult
        """
        budget_tokens = kwargs.get('budget_tokens', 10000)
        max_tokens = kwargs.get('max_tokens', 20000)
        max_retries = kwargs.get('max_retries', 3)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            # "max_tokens": max_tokens,
            # "thinking": {
            #     "type": "enabled",
            #     "budget_tokens": budget_tokens
            # },
            # # Web search tool definition
            # "tools": [{
            #     "type": "function",
            #     "function": {
            #         "name": "web_search",
            #         "description": "Search the web for information",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "query": {
            #                     "type": "string",
            #                     "description": "The search query"
            #                 }
            #             },
            #             "required": ["query"]
            #         }
            #     }
            # }]
            # "thinking": {
            #     "type": "enabled",
            #     "budget_tokens": budget_tokens
            # },
            'reasoning': {
                'max_tokens': budget_tokens  # Allocate specific token count for reasoning
                # Alternatively use 'effort': 'high' for approximate allocation
            },
            "model": self.model,
            "max_tokens": max_tokens,
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 15
            }]
            # 'plugins': [{'id': 'web_search_20250305'}],
        }
        
        return await self.handle_result_with_retries(prompt, payload, max_retries)

    async def deep_research(self, prompt: str, stream: bool = False, **kwargs):
        """
        Perform deep research
        
        Args:
            prompt: The research question or topic
            stream: Whether to stream the response (not implemented)
            **kwargs: Additional keyword arguments
                - max_tokens: Maximum tokens in response (default: 20000)
                - max_retries: Maximum retry attempts (default: 3)
            
        Returns:
            Comprehensive research report as a DeepResearchResult
        """
        # Note: Deep research is not fully supported via OpenRouter's OpenAI-compatible API
        # This is a placeholder implementation
        raise NotImplementedError("Deep research is not fully implemented for Claude via OpenRouter")
    
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

