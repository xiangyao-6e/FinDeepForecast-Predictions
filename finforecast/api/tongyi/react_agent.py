import json
import json5
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
from transformers import AutoTokenizer 
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
import time
import asyncio

from finforecast.api.tongyi.prompt import *
from finforecast.api.tongyi.tool_file import *
from finforecast.api.tongyi.tool_scholar import *
from finforecast.api.tongyi.tool_python import *
from finforecast.api.tongyi.tool_search import *
from finforecast.api.tongyi.tool_visit import *

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

# MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 50))
# TONGYI_API_KEY = os.getenv('TONGYI_API_KEY')
# TONGYI_API_BASE = os.getenv('TONGYI_API_BASE')
from env import TONGYI_API_KEY, TONGYI_API_BASE

TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}

import random
import datetime


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, planning_port, max_tries=20):
        openai_api_key = TONGYI_API_KEY
        openai_api_base = TONGYI_API_BASE

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1 
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model="alibaba/tongyi-deepresearch-30b-a3b",
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=20000,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1)
                )
                content = chat_response.choices[0].message.content
                
                # OpenRouter provides API calling. If you want to use OpenRouter, you need to uncomment line 89 - 90.
                reasoning_content = "<think>\n" + chat_response.choices[0].message.reasoning.strip() + "\n</think>"
                
                content = reasoning_content + content                
                
                if content and content.strip():
                    print("--- Service call successful, received a valid response ---")
                    return content.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except AttributeError as e:
                print(f"Error: Attempt {attempt + 1} failed due to empty result: {e}")
            # except Exception as e:
            #     print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30) 
                
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return f"vllm server error!!!"

    def count_tokens(self, messages):
        tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path) 
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(full_prompt, return_tensors="pt")
        token_count = len(tokens["input_ids"][0])
        
        return token_count

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except: 
            raw_msg = data['item']['messages'][1]["content"] 
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg 

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question
        system_prompt = SYSTEM_PROMPT
        cur_date = today_date()
        system_prompt = system_prompt + str(cur_date)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        while num_llm_calls_available > 0:
            # Check whether time is reached
            if time.time() - start_time > 150 * 60:  # 150 minutes in seconds
                prediction = 'No answer found after 2h30mins'
                termination = 'No answer found after 2h30mins'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages, planning_port)
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                # try:
                if "python" in tool_call.lower():
                    # try:
                    code_raw=content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                    result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                    # except:
                    #     result = "[Python Interpreter Error]: Formatting error."

                else:
                    tool_call = json5.loads(tool_call)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})
                    result = self.custom_call_tool(tool_name, tool_args)

                # except:
                #     result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                # print(result)
                messages.append({"role": "user", "content": result})
            if '<answer>' in content and '</answer>' in content:
                termination = 'answer'
                break
            if num_llm_calls_available <= 0 and '<answer>' not in content:
                messages[-1]['content'] = 'Sorry, the number of llm calls exceeds the limit.'

            max_tokens = 110 * 1024
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = "You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>"
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})
                if '<answer>' in content and '</answer>' in content:
                    prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
                    termination = 'generate an answer as token limit reached'
                else:
                    prediction = messages[-1]['content']
                    termination = 'format error: generate an answer as token limit reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                return result

        if '<answer>' in messages[-1]['content']:
            prediction = messages[-1]['content'].split('<answer>')[1].split('</answer>')[0]
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }
        return result

    def run_with_input(self, question: str, model: str, **kwargs) -> Dict:
        """
        接收一个字符串问题，执行多轮思考和工具调用，并返回最终结果。

        :param question: 用户的输入问题字符串。
        :param model: 要使用的模型标识符。
        :return: 一个包含详细过程和最终预测结果的字典。
        """
        self.model = model
        
        # 1. 初始化变量
        start_time = time.time()
        # planning_port 在 call_server 中未使用，但为保持函数签名一致，可传递一个默认值
        planning_port = 6001  
        # 因为没有评估数据集，所以参考答案（answer）可以设为 N/A
        answer = "N/A"
        self.user_prompt = question
        
        # 2. 构建初始消息
        system_prompt = SYSTEM_PROMPT + str(today_date())
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_count = 0

        # 3. 核心 ReAct 循环 (与 _run 函数完全相同)
        while num_llm_calls_available > 0:
            # 超时检查
            if time.time() - start_time > 150 * 60:  # 150 分钟
                prediction = 'Timeout: No answer found after 2 hours and 30 minutes.'
                termination = 'timeout'
                result = {
                    "question": question, "answer": answer, "messages": messages,
                    "prediction": prediction, "termination": termination
                }
                return result

            round_count += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages, planning_port)
            print(f'Round {round_count}: {content}')

            if '<tool_response>' in content:
                content = content.split('<tool_response>')[0]
            messages.append({"role": "assistant", "content": content.strip()})

            # 工具调用逻辑
            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call_str = content.split('<tool_call>')[1].split('</tool_call>')[0]
                # try:
                if "python" in tool_call_str.lower():
                    # try:
                    code_raw = tool_call_str.split('<code>')[1].split('</code>')[0].strip()
                    tool_result = TOOL_MAP['PythonInterpreter'].call(code_raw)
                    # except Exception:
                    #     tool_result = "[Python Interpreter Error]: Formatting error."
                else:
                    tool_call_obj = json5.loads(tool_call_str)
                    tool_name = tool_call_obj.get('name', '')
                    tool_args = tool_call_obj.get('arguments', {})
                    tool_result = self.custom_call_tool(tool_name, tool_args)
                # except Exception as e:
                #     tool_result = f'Error: Tool call parsing failed. Details: {e}'
                
                tool_result_str = f"<tool_response>\n{tool_result}\n</tool_response>"
                messages.append({"role": "user", "content": tool_result_str})

            # 找到答案，结束循环
            if '<answer>' in content and '</answer>' in content:
                break
            
            # Token 数量检查与处理 (与 _run 逻辑相同)
            max_tokens = 110 * 1024
            token_count = self.count_tokens(messages)
            print(f"Round: {round_count}, Token count: {token_count}")
            
            if token_count > max_tokens:
                print(f"Token limit exceeded: {token_count} > {max_tokens}")
                # 强制模型生成最终答案
                messages[-1]['content'] = ("You have now reached the maximum context length you can handle. You should stop making tool calls and, based on all the information above, think again and provide what you consider the most likely answer in the following format:<think>your final thinking</think>\n<answer>your answer</answer>. You MUST NOT call any more tools, and you MUST follow the specified format and generate the answer in the required language.")
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})
                break  # 强制结束循环

        # 4. 打包最终结果 (与 _run 逻辑相同)
        final_content = messages[-1]['content']
        # if '<answer>' in final_content:
        #     prediction = final_content.split('<answer>')[1].split('</answer>')[0].strip()
        #     termination = 'normal'
        # else:
        #     prediction = 'No answer found within the allowed steps.'
        #     termination = 'max_llm_calls_exceeded'

        # next line for debug
        prediction = final_content
        termination = 'DEBUG'
        result = {
            "question": question,
            "answer": answer, # 参考答案，此处为 "N/A"
            "messages": messages, # 完整的对话历史
            "prediction": prediction, # 模型的最终预测
            "termination": termination # 终止原因
        }
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_args["params"] = tool_args
            if "python" in tool_name.lower():
                result = TOOL_MAP['PythonInterpreter'].call(tool_args)
            elif tool_name == "parse_file":
                params = {"files": tool_args["files"]}
                
                raw_result = asyncio.run(TOOL_MAP[tool_name].call(params, file_root_path="./eval_data/file_corpus"))
                result = raw_result

                if not isinstance(raw_result, str):
                    result = str(raw_result)
            else:
                raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
                result = raw_result
            return result

        else:
            return f"Error: Tool {tool_name} not found"
