import json
import os
import re

from model import ModelClient
from web_tools import WebSearchTool


SYSTEM_PROMPT = """You are a GAIA benchmark agent.

You may use these tools:

web_search(query)
    Search the web. Returns titles, URLs, and snippets.

visit_webpage(url)
    Read the text of a webpage.

At each step, return ONLY valid JSON. Do not wrap it in markdown.
Plain text answers are invalid. Explanations are invalid.

To use a tool:
{
    "type": "tool",
    "tool": "web_search",
    "args": {"query": "..."}
}

or:

{
    "type": "tool",
    "tool": "visit_webpage",
    "args": {"url": "..."}
}

When you have enough evidence:
{
    "type": "final",
    "answer": "..."
}

Continue researching when the available evidence is insufficient.
For questions that do not require tools, answer directly with type="final".
If the question asks about a source, website, article, paper, Wikipedia, a date-specific fact, a URL, a record, a season, or "as of" information, do not answer from memory. Use web_search first.
Search result titles and URLs are not enough evidence for source-based questions. After web_search, use visit_webpage on the most relevant result before finalizing.
Do not return a final answer saying that more research is needed. If more research is needed, call another tool.
The final answer must follow the user's requested format exactly.
If the question asks "how many", the final answer should usually be only the number.
Never include supporting sentences, source notes, or lists unless the user requested them.
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
        self.max_steps = int(os.getenv("AGENT_MAX_STEPS", "8"))

    def run(
        self,
        question: str,
        task_id: str,
        file_name: str | None = None,
    ) -> str:
        """
        Solve one GAIA task and return only the final answer.
        """
        if file_name:
            return self._direct_answer(question)

        history = [
            {
                "role": "user",
                "content": f"Question:\n{question}",
            }
        ]

        last_response = ""
        for _ in range(self.max_steps):
            response = self.model_client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=self._format_history(history),
                json_mode=True,
            )
            last_response = response
            action = self._parse_action(response)

            if action.get("type") == "final":
                return self._clean_answer_for_question(
                    str(action.get("answer", "")),
                    question,
                )

            if action.get("type") == "tool" and self.web_enabled:
                tool_result = self._execute_tool(action)
                history.append({"role": "assistant", "content": response})
                history.append({"role": "tool", "content": tool_result})
                continue

            history.append({"role": "assistant", "content": response})
            history.append(
                {
                    "role": "tool",
                    "content": "Invalid action. Return valid JSON with type='final' or an available tool call.",
                }
            )

        return self._finalize_after_steps(question, history, last_response)

    def _direct_answer(self, question: str) -> str:
        answer = self.model_client.generate(
            system_prompt=(
                "Answer the user's question. Return only the final answer, "
                "with no explanation, no markdown, and no label."
            ),
            user_prompt=question,
        )
        return self._clean_answer_for_question(answer, question)

    def _execute_tool(self, action: dict) -> str:
        tool = action.get("tool")
        args = action.get("args") or {}
        try:
            if tool == "web_search":
                query = str(args.get("query", "")).strip()
                if not query:
                    return "Tool error: missing query."
                results = self.web_search.search(query)
                return self._format_search_results(results)

            if tool == "visit_webpage":
                url = str(args.get("url", "")).strip()
                if not url:
                    return "Tool error: missing url."
                return self.web_search.read_page(url)

            return f"Tool error: unknown tool {tool!r}."
        except Exception as error:
            return f"Tool error: {error}"

    def _format_history(self, history: list[dict]) -> str:
        return "\n\n".join(
            f"{item['role'].upper()}:\n{item['content']}" for item in history
        )

    def _format_search_results(self, results) -> str:
        if not results:
            return "No search results found."
        lines = []
        for index, result in enumerate(results, start=1):
            snippet = f"\nSnippet: {result.snippet}" if result.snippet else ""
            lines.append(f"[{index}] {result.title}\nURL: {result.url}{snippet}")
        return "\n\n".join(lines)

    def _parse_action(self, response: str) -> dict:
        text = response.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()
        try:
            action = json.loads(text)
        except json.JSONDecodeError:
            return {"type": "invalid", "raw": response}
        if not isinstance(action, dict):
            return {"type": "invalid", "raw": response}
        return action

    def _finalize_after_steps(
        self,
        question: str,
        history: list[dict],
        last_response: str,
    ) -> str:
        answer = self.model_client.generate(
            system_prompt=(
                "Return the best exact final answer to the user's original question "
                "using the conversation and tool results. Return only the answer."
            ),
            user_prompt=f"Original question:\n{question}\n\nHistory:\n{self._format_history(history)}",
        )
        cleaned = self._clean_answer_for_question(answer, question)
        return cleaned or self._clean_answer(last_response)

    def _clean_answer_for_question(self, answer: str, question: str) -> str:
        cleaned = self._clean_answer(answer)
        if question.strip().lower().startswith("how many"):
            match = re.search(r"\b\d+\b", cleaned)
            if match:
                return match.group(0)
        return cleaned

    def _clean_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        prefixes = ("FINAL ANSWER:", "Final answer:", "Answer:", "ANSWER:")
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned.strip().strip('"').strip()
