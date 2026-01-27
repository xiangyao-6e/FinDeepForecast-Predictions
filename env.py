import os
import pathlib

from dotenv import load_dotenv

env_file = pathlib.Path(__file__).parent / '.env'

load_dotenv(env_file)

OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')
GROK_API_KEY = os.environ.get('GROK_API_KEY')

OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

TRACE_ENABLED = os.environ.get('TRACE_ENABLED') == 'true'
CACHE_OVERWRITE = os.environ.get('CACHE_OVERWRITE', 'false') == 'true'
CACHE_DISABLED = os.environ.get('CACHE_DISABLED', 'true') == 'true'
CACHE_ONLY = os.environ.get('CACHE_ONLY', 'false') == 'true'

# Tongyi Keys and Configurations
SERPER_KEY_ID = os.environ.get('SERPER_KEY_ID')
JINA_API_KEYS = os.environ.get('JINA_API_KEYS')

TONGYI_API_KEY = os.environ.get('TONGYI_API_KEY', OPENROUTER_API_KEY)
TONGYI_API_BASE = os.environ.get("TONGYI_API_BASE", "https://openrouter.ai/api/v1")
SUMMARY_MODEL_NAME = os.environ.get("SUMMARY_MODEL_NAME", "qwen/qwen3-30b-a3b-thinking-2507")
USE_IDP = os.environ.get("USE_IDP", False)

SANDBOX_FUSION_ENDPOINTS = os.environ.get("SANDBOX_FUSION_ENDPOINTS", "http://localhost:8080/")