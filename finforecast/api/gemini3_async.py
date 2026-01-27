#!/usr/bin/env python3
"""
OpenAI API Script with GPT-5 Model, Thinking, and Search Functions

This script provides a comprehensive interface to OpenAI's API with:
- GPT-5 model integration
- Thinking function for step-by-step reasoning
- Search function for information retrieval
- Error handling and logging
- Configurable parameters
"""

"""
from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-pro-preview",
    contents="Find the race condition in this multi-threaded C++ snippet: [code here]",
)

print(response.text)
"""
import time
import asyncio
# from typing import Dict, List, Optional, Any, Self
from google import genai


from env import GEMINI_API_KEY

from finforecast.log_util import get_logger
from finforecast.objs import Citation, DeepResearchResult
from finforecast.api.base_api import BaseAPI

from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch, ThinkingConfig

from pdb import set_trace

logger = get_logger()

API_KEY = GEMINI_API_KEY

## TODO: Need to implement thinking
class AsyncGemini3API(BaseAPI):
    def __init__(self, function: str):
        self.function = function
        self.api_key = API_KEY
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-3-pro-preview"
        self.tools =  [Tool(google_search = GoogleSearch())]

    def handle_non_stream_output(self,response):
        '''
        return final_output, citations, reasoning_steps, web_searches, response
        '''
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


    def handle_stream_output(self,response):
        raise NotImplementedError("Stream output is not implemented yet")


    def handle_response(self, prompt: str, response, stream: bool):
        if not stream:
            final_output, citations, reasoning_steps, web_searches = self.handle_non_stream_output(response)
        else:
            final_output, citations, reasoning_steps, web_searches, response = self.handle_stream_output(response)
        start = time.time()
        result = DeepResearchResult(
            prompt=prompt,
            report=final_output,
            citations=citations,
            reasoning_steps=reasoning_steps,
            web_searches=web_searches,
            raw_response=response
        )
        duration = time.time() - start
        result.time = duration
        # result.pretty_print(logger)
        logger.info(
            f'✅ Deep Research completed: {len(citations)} citations, {len(reasoning_steps)} reasoning steps'
        )
        return result
    
    async def handle_result_with_retries(self, prompt: str, payload, stream: bool, max_retries: int):
        for attempt in range(max_retries):
            logger.info(f"Attempt {attempt + 1} of {max_retries} for model {self.model} function {self.function} request")
            start = time.time()
            # Run the synchronous API call in a thread pool to make it async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.models.generate_content(**payload)
            )
            # from pdb import set_trace; set_trace()
            result = self.handle_response(prompt, response.model_dump(warnings=False), stream)
            # set_trace()
            duration = time.time() - start
            result.time = duration
            logger.info(f"Function {self.function} process completed successfully")
            return result

    async def thinking(self, prompt: str, stream: bool, **kwargs):
        payload = {
            'model': self.model,
            'contents': prompt,
            'config': GenerateContentConfig(
                thinking_config = ThinkingConfig(
                    thinking_budget = -1,
                    include_thoughts = True
                )
            )
        }
        max_retries = kwargs.get('max_retries', 3)
        return await self.handle_result_with_retries(prompt, payload, stream, max_retries)

    async def thinking_and_search(self, prompt: str, stream: bool, **kwargs):
        payload = {
            'model': self.model,
            'contents': prompt,
            'config': GenerateContentConfig(
                tools=self.tools,
                thinking_config = ThinkingConfig(
                    thinking_budget = -1,
                    include_thoughts = True
                )
            )
        }
        max_retries = kwargs.get('max_retries', 3)
        return await self.handle_result_with_retries(prompt, payload, stream, max_retries)

    
    async def __call__(self, prompt: str, stream: bool = False, **kwargs):
        if self.function == "thinking":
            return await self.thinking(prompt, stream, **kwargs)
        elif self.function == "thinking_and_search":
            return await self.thinking_and_search(prompt, stream, **kwargs)
        elif self.function == "deep_research":
            return await self.deep_research(prompt, stream, **kwargs)




