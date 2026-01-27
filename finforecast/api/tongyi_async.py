#!/usr/bin/env python3
"""
Async Grok API Script with Grok-4 Model, Thinking, and Search Functions

This script provides an async interface to Grok's API with:
- Grok-4 model integration
- Thinking function for step-by-step reasoning
- Search function for information retrieval
- Error handling and logging
- Configurable parameters
- Async/await patterns for non-blocking operations
"""

import asyncio
import aiohttp

from finforecast.log_util import get_logger
from finforecast.objs import Citation, DeepResearchResult
from finforecast.api.base_api import BaseAPI
from finforecast.api.tongyi.react_agent import MultiTurnReactAgent

import re
from pdb import set_trace

logger = get_logger()

class AsyncTongyiAPI(BaseAPI):
    """Async main class for Grok API operations with thinking and search capabilities"""
    
    def __init__(self, function: str):
        """
        Initialize the async Grok script
        
        Args:
            function: The function type to use ("thinking", "thinking_and_search", "deep_research")
        """

        assert function in ["thinking", "thinking_and_search", "deep_research"]
        self.function = function

        # Initialize Agent
        llm_cfg = {
            'model': 'Alibaba-NLP/Tongyi-DeepResearch-30B-A3B',  # For tokenizer path
            'generate_cfg': {
                'temperature': 0.6,
                'top_p': 0.95,
                'presence_penalty': 1.1
            }
        }
        self.agent = MultiTurnReactAgent(llm=llm_cfg)
        self.model = "alibaba/tongyi-deepresearch-30b-a3b"

    async def handle_result_with_retries(self, prompt: str, max_retries: int):
        def extract_answer_block(prediction_text: str):
            """
            Extract only the content between <answer> and </answer> tags.
            
            Args:
                prediction_text: The full prediction text containing <think> and <answer> blocks
            
            Returns:
                Extracted answer text (all answer blocks concatenated with newlines), 
                or the full text if no answer tags found
            """
            if not prediction_text:
                return ""
            
            # Try to extract all content between <answer> and </answer>
            answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
            matches = re.findall(answer_pattern, prediction_text, re.DOTALL | re.IGNORECASE)
            
            if matches:
                # Join all answer blocks with newlines
                return [match.strip() for match in matches][-1]
            else:
                # If no answer tags found, return the original text
                # (in case the format is different or for error messages)
                return prediction_text.strip()
        
        """Handle API request with retries and exponential backoff"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} of {max_retries} for model {self.model} function {self.function} request")
                # start = asyncio.get_event_loop().time()
                loop = asyncio.get_event_loop()
                start = loop.time()
                response_json = await loop.run_in_executor(
                    None,
                    lambda: self.agent.run_with_input(question=str(prompt), model=self.model)
                )
                
                # async with aiohttp.ClientSession() as session:
                #     async with session.post(
                #         self.url, 
                #         headers=self.headers, 
                #         json=payload,
                #         timeout=aiohttp.ClientTimeout(total=3600)
                #     ) as response:
                #         response_json = await response.json()
                
                # final_output = response_json['choices'][0]['message']['content']
                # # logger.info(f"Final Output:\n {final_output}")
                # citations = []
                # if response_json.get('citations'):
                #     citations = [Citation(
                #                 title="",
                #                 url=url) for url in response_json['citations']]

                final_answer = extract_answer_block(response_json['prediction'])
                duration = asyncio.get_event_loop().time() - start
                result = DeepResearchResult(
                    prompt=prompt,
                    report=final_answer,
                    citations=[],
                    reasoning_steps=[],
                    web_searches=[],
                    raw_response=response_json,
                    time=duration
                )
                # result.pretty_print(logger)
                logger.info(f"Function {self.function} process completed successfully")
                return result
            except KeyError as e:
                # logger.error(f"KeyError in thinking process (attempt {attempt + 1}): {str(e)}")
                # if attempt == max_retries - 1:
                #     # Last attempt failed, raise the exception
                #     logger.error(f"All {max_retries} attempts failed for function {self.function} request")
                #     raise e
                # else:
                #     # Wait before retrying (exponential backoff)
                #     wait_time = 2 ** attempt
                #     logger.info(f"Retrying in {wait_time} seconds...")
                #     await asyncio.sleep(wait_time)
                raise Exception("Insufficient balance for the Tongyi API Key!")
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

    async def deep_research(self, prompt: str, stream: bool, **kwargs):
        """
        Deep research is not implemented for Grok API yet
        
        Args:
            prompt: The research question or topic
            stream: Whether to stream the response
            
        Raises:
            NotImplementedError: This function is not yet supported by Grok API
        """
        # raise NotImplementedError("Deep research is not implemented yet")
        max_retries = kwargs.get("max_retries", 3)
        return await self.handle_result_with_retries(prompt, max_retries)
    
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

