class BaseAPI:
    def thinking(self, prompt: str, **kwargs):
        raise NotImplementedError("Thinking is not implemented yet")

    def thinking_and_search(self, prompt: str, **kwargs):
        raise NotImplementedError("Thinking and search is not implemented yet")

    def deep_research(self, prompt: str, **kwargs):
        raise NotImplementedError("Deep research is not implemented yet")