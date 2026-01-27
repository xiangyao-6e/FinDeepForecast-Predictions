#!/usr/bin/env python3
"""
Async Gemini API Script with Gemini 2.5 Pro Model, Thinking, and Search Functions

This script provides an async interface to Google's Gemini API with:
- Gemini 2.5 Pro model integration
- Thinking function for step-by-step reasoning
- Search function for information retrieval
- Error handling and logging
- Configurable parameters
- Async/await patterns for non-blocking operations
"""
import asyncio
# from typing import Dict, List, Optional, Any, Self

from env import GEMINI_API_KEY
from finforecast.log_util import get_logger
from finforecast.objs import Citation, DeepResearchResult
from finforecast.api.base_api import BaseAPI

from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, ThinkingConfig

from pdb import set_trace

logger = get_logger()

API_KEY = GEMINI_API_KEY

class AsyncGeminiAPI(BaseAPI):
    """Async main class for Gemini API operations with thinking and search capabilities"""
    
    def __init__(self, function: str):
        """
        Initialize the async Gemini script
        
        Args:
            function: The function type to use ("thinking", "thinking_and_search", "deep_research")
        """
        assert function in ["thinking", "thinking_and_search", "deep_research"]
        self.function = function
        self.api_key = API_KEY
        # Note: Using regular client as Google's genai library handles async internally
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-pro"
        self.tools = [Tool(google_search=GoogleSearch())]

    def handle_non_stream_output(self, response):
        """
        Handle non-streaming output
        
        Returns:
            tuple: (final_output, citations, reasoning_steps, web_searches)
        """
        final_output = ""
        for part in response['candidates'][0]['content']['parts']:
            if part.get('thought') is None and part.get('text'):
                final_output += part['text']
        
        # extract citations
        citations = []
        
        if 'grounding_metadata' in response['candidates'][0]:
            grounding_metadata = response['candidates'][0].get('grounding_metadata', {})
            if grounding_metadata and grounding_metadata.get('grounding_chunks'):
                grounding_chunks = grounding_metadata.get('grounding_chunks', [])
                if grounding_chunks:
                    for chunk in grounding_chunks:
                        if chunk.get('web', {}).get('title'):
                            citations.append(Citation(
                                title=chunk['web']['title'],
                                url=chunk['web']['uri']
                            ))
        
        # extract reasoning steps
        reasoning_steps = []
        content = response['candidates'][0].get("content", {})
        parts = content.get('parts', [])
        for part in parts:
            if part.get('thought') == True and part.get('text'):
                reasoning_steps.append(part['text'])

        # extract web searches
        web_searches = []
        if 'grounding_metadata' in response['candidates'][0]:
            grounding_metadata = response['candidates'][0].get('grounding_metadata', {})
            if grounding_metadata and grounding_metadata.get('web_search_queries'):
                web_search_queries = grounding_metadata.get('web_search_queries', [])
                if web_search_queries:
                    for query in web_search_queries:
                        if query:
                            web_searches.append(query)
        
        return final_output, citations, reasoning_steps, web_searches

    async def handle_stream_output(self, response):
        """Stream output handler - not implemented yet"""
        raise NotImplementedError("Stream output is not implemented yet")

    async def handle_response(self, prompt: str, response, stream: bool):
        """
        Handle the response from Gemini API
        
        Args:
            prompt: The original prompt
            response: The API response
            stream: Whether streaming was enabled
            
        Returns:
            DeepResearchResult object
        """
        if not stream:
            final_output, citations, reasoning_steps, web_searches = self.handle_non_stream_output(response)
        else:
            final_output, citations, reasoning_steps, web_searches = await self.handle_stream_output(response)
        
        result = DeepResearchResult(
            prompt=prompt,
            report=final_output,
            citations=citations,
            reasoning_steps=reasoning_steps,
            web_searches=web_searches,
            raw_response=response
        )
        # result.pretty_print(logger)
        logger.info(
            f'✅ Gemini Research completed: {len(citations)} citations, {len(reasoning_steps)} reasoning steps'
        )
        return result
    
    async def handle_result_with_retries(self, prompt: str, payload, stream: bool, max_retries: int):
        """
        Handle API request with retries and exponential backoff
        
        Args:
            prompt: The prompt text
            payload: The API payload
            stream: Whether to stream responses
            max_retries: Maximum number of retry attempts
            
        Returns:
            DeepResearchResult object
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} of {max_retries} for model {self.model} function {self.function} request")
                start = asyncio.get_event_loop().time()
                
                # Run the synchronous API call in a thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(**payload)
                )
                
                result = await self.handle_response(prompt, response.model_dump(warnings=False), stream)
                duration = asyncio.get_event_loop().time() - start
                result.time = duration
                logger.info(f"Function {self.function} process completed successfully in {duration:.2f}s")
                return result
            except Exception as e:
                logger.error(f"Error in thinking process (attempt {attempt + 1}): {str(e)}")
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
        Perform step-by-step thinking and reasoning
        
        Args:
            prompt: The problem or question to think about
            stream: Whether to stream the response (not implemented)
            
        Returns:
            Detailed thinking process as a DeepResearchResult
        """
        payload = {
            'model': self.model,
            'contents': prompt,
            'config': GenerateContentConfig(
                thinking_config=ThinkingConfig(
                    thinking_budget=-1,
                    include_thoughts=True
                )
            )
        }
        max_retries = kwargs.get('max_retries', 3)
        return await self.handle_result_with_retries(prompt, payload, stream, max_retries)

    async def thinking_and_search(self, prompt: str, stream: bool, **kwargs):
        """
        Perform thinking with web search capabilities
        
        Args:
            prompt: The problem or question to think about
            stream: Whether to stream the response (not implemented)
            
        Returns:
            Detailed thinking process with search results as a DeepResearchResult
        """
        payload = {
            'model': self.model,
            'contents': prompt,
            'config': GenerateContentConfig(
                tools=self.tools,
                thinking_config=ThinkingConfig(
                    thinking_budget=-1,
                    include_thoughts=True
                )
            )
        }
        max_retries = kwargs.get('max_retries', 3)
        return await self.handle_result_with_retries(prompt, payload, stream, max_retries)

    async def deep_research(self, prompt: str, stream: bool = False, **kwargs):
        """
        Deep research is not implemented for Gemini API yet
        
        Args:
            prompt: The research question or topic
            stream: Whether to stream the response
            
        Raises:
            NotImplementedError: This function is not yet supported by Gemini API
        """
        raise NotImplementedError("Deep research is not implemented yet")
    
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

