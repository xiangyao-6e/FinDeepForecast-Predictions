import logging
from typing import Any, List, Optional

from pydantic import BaseModel

from finforecast.log_util import pretty_log_research_result


class Citation(BaseModel):
    """Citation from Deep Research API"""
    title: Optional[str] = None
    url: Optional[str] = None

    # openai
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    excerpt: Optional[str] = None

    # qwen
    index_number: Optional[int] = None
    description: Optional[str] = None

    def __str__(self):
        if self.title is None:
            return f"{self.url}"
        if self.url is None:
            return f"{self.title}"
        return f"{self.url}: {self.title}"


class DeepResearchResult(BaseModel):
    """Result from Deep Research API"""
    prompt: str = ''
    report: str = ''
    citations: List[Citation] = []
    reasoning_steps: List[str] = []
    web_searches: List[str] = []
    raw_response: Any = None
    time: float = 0.0

    def pretty_print(self, logger: logging.Logger):
        pretty_log_research_result(
            logger, self.report, self.web_searches, self.reasoning_steps, self.citations
        )
