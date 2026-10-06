import os

from model import ModelClient
from web_tools import WebSearchTool


SYSTEM_PROMPT = """You are a GAIA benchmark answer agent.
Answer the user's question using only your reasoning and prior knowledge.
These tasks often include simple transformations, reversed text, tables, wordplay, and strict output formats.
Decode or transform the question when needed, reason silently, and then answer.
Return only the final answer, with no explanation, no markdown, and no label.
If the question requests a specific format, follow it exactly.
"""

QUERY_PROMPT = """Create one concise web search query to answer the user's question.
Return only the search query, with no quotes and no explanation.
"""

WEB_ANSWER_PROMPT = """You are a GAIA benchmark answer agent.
Answer the user's question using the supplied web evidence.
Return only the final answer, with no explanation, no markdown, and no label.
If the evidence is insufficient, use your best judgment, but keep the answer concise.
"""


class GaiaAgent:
    def __init__(
        self,
        model_client: ModelClient | None = None,
        web_search: WebSearchTool | None = None,
    ):
        self.model_client = model_client or ModelClient()
        self.web_search = web_search or WebSearchTool()
        self.web_enabled = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"

    def run(
        self,
        question: str,
        task_id: str,
        file_name: str | None = None,
    ) -> str:
        """
        Solve one GAIA task and return only the final answer.
        """
        if self.web_enabled and not file_name and self._should_use_web(question):
            web_answer = self._try_web_answer(question)
            if web_answer:
                return web_answer

        answer = self.model_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=question,
        )
        return self._clean_answer(answer)

    def _try_web_answer(self, question: str) -> str:
        try:
            query = self.model_client.generate(
                system_prompt=QUERY_PROMPT,
                user_prompt=question,
            )
            query = self._clean_answer(query) or question
            evidence = self.web_search.gather_evidence(query)
            if not evidence.strip():
                return ""
            answer = self.model_client.generate(
                system_prompt=WEB_ANSWER_PROMPT,
                user_prompt=f"Question:\n{question}\n\nWeb evidence:\n{evidence}",
            )
            return self._clean_answer(answer)
        except Exception as error:
            print(f"Web answer failed, falling back to base model: {error}")
            return ""

    def _should_use_web(self, question: str) -> bool:
        lowered = question.lower()
        web_signals = (
            "http://",
            "https://",
            "wikipedia",
            "latest",
            "as of",
            "published",
            "article",
            "paper",
            "record",
            "season",
            "olympics",
            "competition",
            "universe today",
            "libretext",
            "wikidata",
        )
        return any(signal in lowered for signal in web_signals)

    def _clean_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        prefixes = ("FINAL ANSWER:", "Final answer:", "Answer:", "ANSWER:")
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned.strip().strip('"').strip()
