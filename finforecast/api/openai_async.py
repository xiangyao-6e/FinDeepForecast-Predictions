#!/usr/bin/env python3
"""
Async OpenAI API Script with GPT-5 Model, Thinking, and Search Functions

This script provides an async interface to OpenAI's API with:
- GPT-5 model integration
- Thinking function for step-by-step reasoning
- Search function for information retrieval
- Error handling and logging
- Configurable parameters
- Async/await patterns for non-blocking operations
"""

import asyncio

from env import OPENAI_KEY
from openai import AsyncOpenAI
# from finforecast.prompt_util import generate_prompt

from typing_extensions import Iterable
from finforecast.log_util import get_logger
from finforecast.objs import Citation, DeepResearchResult
from finforecast.api.base_api import BaseAPI

from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_code_interpreter_tool_call import (
    ResponseCodeInterpreterToolCall,
)
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_function_web_search import (
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from pdb import set_trace

logger = get_logger()

API_KEY = OPENAI_KEY

class AsyncOpenAIAPI(BaseAPI):
    """Async main class for OpenAI API operations with thinking and search capabilities"""
    
    def __init__(self, function: str, model: str = "gpt-5-2025-08-07"):
        """
        Initialize the async OpenAI script
        
        Args:
            function: The function type to use ("thinking", "thinking_and_search", "deep_research")
        """

        assert function in ["test_mode", "thinking", "thinking_and_search", "deep_research"]
        self.function = function
        self.api_key = API_KEY
        
        self.client = AsyncOpenAI(api_key=self.api_key, timeout=3600.0)
        self.model = model
        
    async def handle_non_stream_output(self, response):
        """Handle non-streaming output"""
        # Extract the final output text
        final_output = response.output[-1].content[0].text

        # Extract citations
        citations = []
        if hasattr(response.output[-1].content[0], 'annotations'):
            for annotation in response.output[-1].content[0].annotations:
                if hasattr(annotation, 'start_index'):
                    # Extract excerpt from the text
                    start = annotation.start_index
                    end = annotation.end_index
                    excerpt = final_output[start:end] if start < len(final_output) else ""

                    citations.append(
                        Citation(
                            title=getattr(annotation, 'title', 'Unknown Title'),
                            url=getattr(annotation, 'url', ''),
                            start_index=start,
                            end_index=end,
                            excerpt=excerpt,
                        )
                    )

        # Extract reasoning steps
        reasoning_steps = []
        for item in response.output:
            if item.type == 'reasoning' and hasattr(item, 'summary'):
                for summary_item in item.summary:
                    if hasattr(summary_item, 'text'):
                        reasoning_steps.append(summary_item.text)

        # Extract web search queries
        web_searches = []
        for item in response.output:
            if item.type == 'web_search_call' and hasattr(item, 'action'):
                if hasattr(item.action, 'query') and item.action.query:
                    web_searches.append(item.action.query)

        return final_output, citations, reasoning_steps, web_searches

    async def handle_stream_output(self, stream: Iterable[ResponseStreamEvent]):
        """Stream deep research results to Streamlit UI components"""

        reasoning_steps = []
        final_output = ''
        web_searches = []
        citations = []

        async for event in stream:
            # Handle reasoning summary text completion
            if isinstance(event, ResponseReasoningSummaryTextDoneEvent):
                logger.info(event.text)

            # Handle text delta events (main answer content)
            elif isinstance(event, ResponseTextDeltaEvent):
                pass

            # Handle response creation and completion
            elif isinstance(event, ResponseCreatedEvent):
                logger.info(event.response.output_text)

            elif isinstance(event, ResponseCompletedEvent):
                logger.info('✅ Research completed!')
                logger.info(event.response.usage.model_dump())
                return await self.handle_non_stream_output(event.response) + (event.response,)

            # Handle tool calls and web search activities
            elif isinstance(
                event, (ResponseOutputItemDoneEvent, ResponseOutputItemAddedEvent)
            ):
                if isinstance(event.item, ResponseFunctionWebSearch):
                    try:
                        item_dict = event.item.to_dict(warnings=False)
                        if 'action' in item_dict and isinstance(item_dict['action'], dict):
                            action_dict = item_dict['action']
                            action_type = action_dict.get('type')

                            if action_type == 'search':
                                query = action_dict.get('query', '')
                                current_activity = f'**Search:** {query}'

                                logger.info(f'{current_activity}')

                            elif action_type == 'open_page':
                                url = action_dict.get('url', '')
                                if url:
                                    current_activity = f'**Reading page:** {url}'
                                    logger.info(f'{current_activity}')

                            elif action_type == 'find_in_page':
                                search_pattern = action_dict.get('pattern', '').strip()
                                url = action_dict.get('url', '')
                                if url:
                                    if search_pattern:
                                        current_activity = (
                                            f'**Page search:** "{search_pattern}" on {url}'
                                        )
                                        logger.info(f'{current_activity}')
                                    else:
                                        current_activity = f'**Reading page:** {url}'
                                        logger.info(f'{current_activity}')

                    except Exception as e:
                        logger.info(f'Error processing web search event: {e}')

                # Handle other tool call types
                elif isinstance(
                    event.item,
                    (
                        ResponseOutputMessage,
                        ResponseReasoningItem,
                        ResponseFunctionToolCall,
                        ResponseCodeInterpreterToolCall,
                        ResponseOutputTextAnnotationAddedEvent,
                    ),
                ):
                    # These are handled but don't need special UI updates
                    # logger.info(f'Unhandled item type: {event.item.type} {event.item.model_dump()}')
                    pass
                else:
                    logger.info(f'Unhandled item type: {event.item.type} {event.item.model_dump()}')
            else:
                # pass
                logger.info(f'Unhandled item type: {event.item.type} {event.item.model_dump()}')

        return final_output, citations, reasoning_steps, web_searches, None

    async def handle_response(self, prompt: str, response, stream: bool):
        if not stream:
            final_output, citations, reasoning_steps, web_searches = await self.handle_non_stream_output(response)
        else:
            final_output, citations, reasoning_steps, web_searches, response = await self.handle_stream_output(response)
        
        result = DeepResearchResult(
            prompt=prompt,
            report=final_output,
            citations=citations,
            reasoning_steps=reasoning_steps,
            web_searches=web_searches,
            raw_response=response.model_dump(warnings=False)
        )
        # result.pretty_print(logger)
        logger.info(
            f'✅ OpenAI {self.function} completed: {len(citations)} citations, {len(reasoning_steps)} reasoning steps'
        )
        return result
    
    async def handle_result_with_retries(self, prompt: str, payload, stream: bool, max_retries: int):
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1} of {max_retries} for model {self.model} function {self.function} request")
                start = asyncio.get_event_loop().time()
                response = await self.client.responses.create(
                    **payload
                )
                result = await self.handle_response(prompt, response, stream)
                duration = asyncio.get_event_loop().time() - start
                result.time = duration
                logger.info(f"Function {self.function} process completed successfully")
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

    async def test_mode(self, prompt: str, stream: bool, **kwargs):
        """
        Perform step-by-step thinking and reasoning
        
        Args:
            prompt: The problem or question to think about
            stream: Whether to stream the response
            
        Returns:
            Detailed thinking process as a DeepResearchResult
        """
        payload = {
            'model': self.model,
            # 'reasoning': {'effort': 'high'},
            'input': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.0,
        }
        max_retries = kwargs.get('max_retries', 3)
        return await self.handle_result_with_retries(prompt, payload, stream, max_retries)
        
    async def thinking(self, prompt: str, stream: bool, **kwargs):
        """
        Perform step-by-step thinking and reasoning
        
        Args:
            prompt: The problem or question to think about
            stream: Whether to stream the response
            
        Returns:
            Detailed thinking process as a DeepResearchResult
        """
        payload = {
            'model': self.model,
            'reasoning': {'effort': 'high'},
            'input': [
                {'role': 'user', 'content': prompt}
            ],
        }
        max_retries = kwargs.get('max_retries', 3)
        return await self.handle_result_with_retries(prompt, payload, stream, max_retries)
    
    async def thinking_and_search(self, prompt: str, stream: bool, **kwargs):
        """
        Perform thinking with web search capabilities
        
        Args:
            prompt: The problem or question to think about
            stream: Whether to stream the response
            
        Returns:
            Detailed thinking process with search results as a DeepResearchResult
        """
        tools = kwargs.get('tools', [
            {'type': 'web_search'},
            # {'type': 'code_interpreter', 'container': {'type': 'auto', 'file_ids': []}}
        ])

        reasoning_config = {'effort': 'medium', 'summary': 'auto'}

        payload = {
            'model': self.model, 'input': prompt, 'tools': tools, 'reasoning': reasoning_config, 'stream': stream
        }
        
        max_retries = kwargs.get('max_retries', 3)
        return await self.handle_result_with_retries(prompt, payload, stream, max_retries)

    async def deep_research(self, prompt: str, stream: bool = True, **kwargs):
        """
        Perform deep research using OpenAI's Deep Research API
        
        Args:
            prompt: The research question or topic
            stream: Whether to stream the response
            **kwargs: Additional keyword arguments including:
                - system_message: Optional system message for research context
                - tools: Custom tools configuration
                - model: Model to use (defaults to o3-deep-research-2025-06-26)
            
        Returns:
            DeepResearchResult with report, citations, reasoning steps, and web searches
        """
        # logger.info(f'🔍 Starting Deep Research API query: {prompt[:100]}...')

        # Get system message and model from kwargs
        system_message = kwargs.get('system_message', '')
        model = kwargs.get('model', 'o3-deep-research-2025-06-26')

        # Prepare the input messages
        input_messages = []

        if system_message:
            input_messages.append({
                'role': 'developer',
                'content': [{'type': 'input_text', 'text': system_message}]
            })

        input_messages.append({
            'role': 'user',
            'content': [{'type': 'input_text', 'text': prompt}]
        })

        # Configure tools - always include web search
        tools = kwargs.get('tools', [
            {'type': 'web_search'},
            {'type': 'code_interpreter', 'container': {'type': 'auto', 'file_ids': []}}
        ])

        # only medium is supported for o3-deep & o4-mini-deep
        reasoning_config = {'effort': 'medium', 'summary': 'auto'}

        try:
            # Make the actual Deep Research API call
            request_params = {
                'model': model, 
                'input': input_messages, 
                'tools': tools, 
                'reasoning': reasoning_config, 
                'stream': stream
            }

            # Try with reasoning first, fall back without if organization not verified
            response = await self.client.responses.create(**request_params)
            logger.info('✅ Deep Research with reasoning enabled')

            if not stream:
                final_output, citations, reasoning_steps, web_searches = await self.handle_non_stream_output(response)
            else:
                final_output, citations, reasoning_steps, web_searches, response = await self.handle_stream_output(response)

            result = DeepResearchResult(
                prompt=prompt,
                report=final_output,
                citations=citations,
                reasoning_steps=reasoning_steps,
                web_searches=web_searches,
                raw_response=response.model_dump(warnings=False)
            )
            result.pretty_print(logger)
            logger.info(
                f'✅ Deep Research completed: {len(citations)} citations, {len(reasoning_steps)} reasoning steps'
            )
            return result

        except Exception as e:
            logger.error(f'❌ Deep Research API error: {e}', exc_info=True)
            return DeepResearchResult()

    async def __call__(self, prompt: str, stream: bool = False, **kwargs):
        if self.function == "thinking":
            return await self.thinking(prompt, stream, **kwargs)
        elif self.function == "test_mode":
            return await self.test_mode(prompt, stream, **kwargs)
        elif self.function == "thinking_and_search":
            return await self.thinking_and_search(prompt, stream, **kwargs)
        elif self.function == "deep_research":
            return await self.deep_research(prompt, stream, **kwargs)


# # Batch processing helper functions
# def create_task(company_name: str, model: str = 'o3-deep-research-2025-06-26', custom_prompt: str = ''):
#     """
#     Create a batch task for deep research
    
#     Args:
#         company_name: Name of the company for research
#         model: Model to use for research
#         custom_prompt: Custom prompt, if not provided will generate from company name
        
#     Returns:
#         Task dictionary for batch processing
#     """
#     if not custom_prompt:
#         prompt = generate_prompt(company=company_name)
#     else:
#         prompt = custom_prompt
#     return {
#         'custom_id': company_name,
#         'method': 'POST',
#         'url': '/v1/responses',
#         'body': {
#             'model': model,
#             'tools': [
#                 {'type': 'web_search_preview'},
#                 {'type': 'code_interpreter', 'container': {'type': 'auto', 'file_ids': []}},
#             ],
#             'reasoning': {'effort': 'medium', 'summary': 'auto'},
#             'input': [{'role': 'user', 'content': prompt}],
#         },
#     }


# async def submit_batch_tasks(tasks: List[dict], save_path: str, api_key: str = API_KEY):
#     """
#     Submit batch tasks for processing
    
#     Args:
#         tasks: List of task dictionaries
#         save_path: Path to save the batch task file
#         api_key: OpenAI API key
        
#     Returns:
#         None
#     """
#     client = AsyncOpenAI(api_key=api_key, timeout=3600.0)
    
#     # Create a JSONL string in memory
#     buffer = io.BytesIO()
#     for obj in tasks:
#         line = json.dumps(obj) + '\n'
#         buffer.write(line.encode('utf-8'))

#     # Reset cursor to the beginning so the API can read from start
#     buffer.seek(0)

#     batch_file = await client.files.create(file=buffer, purpose='batch')

#     logger.info(batch_file)

#     pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
#     pathlib.Path(save_path).write_bytes(buffer.getvalue())

#     batch_job = await client.batches.create(
#         input_file_id=batch_file.id, endpoint='/v1/responses', completion_window='24h'
#     )
#     batch_job = await client.batches.retrieve(batch_job.id)
#     logger.info(batch_job)

# def handle_non_stream_output_standalone(response):
#     """Handle non-streaming output (standalone version for batch processing)"""
#     # Extract the final output text
#     final_output = response.output[-1].content[0].text

#     # Extract citations
#     citations = []
#     if hasattr(response.output[-1].content[0], 'annotations'):
#         for annotation in response.output[-1].content[0].annotations:
#             if hasattr(annotation, 'start_index'):
#                 # Extract excerpt from the text
#                 start = annotation.start_index
#                 end = annotation.end_index
#                 excerpt = final_output[start:end] if start < len(final_output) else ""

#                 citations.append(
#                     Citation(
#                         title=getattr(annotation, 'title', 'Unknown Title'),
#                         url=getattr(annotation, 'url', ''),
#                         start_index=start,
#                         end_index=end,
#                         excerpt=excerpt,
#                     )
#                 )

#     # Extract reasoning steps
#     reasoning_steps = []
#     for item in response.output:
#         if item.type == 'reasoning' and hasattr(item, 'summary'):
#             for summary_item in item.summary:
#                 if hasattr(summary_item, 'text'):
#                     reasoning_steps.append(summary_item.text)

#     # Extract web search queries
#     web_searches = []
#     for item in response.output:
#         if item.type == 'web_search_call' and hasattr(item, 'action'):
#             if hasattr(item.action, 'query') and item.action.query:
#                 web_searches.append(item.action.query)

#     return final_output, citations, reasoning_steps, web_searches
