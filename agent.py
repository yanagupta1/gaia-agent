import json
import os
import re

from file_tools import AttachmentContext, AttachmentTool
from model import ModelClient
from multimodal_tools import MultimodalTool
from web_tools import WebSearchTool


SYSTEM_PROMPT = """You are a GAIA benchmark agent. Respond with ONLY valid JSON (no markdown, no prose).

Tools:
- web_search(query): web results (title, url, snippet)
- visit_webpage(url): read a page's text
- attachment_summary() / read_attachment_text(): inspect or read an attached file
- execute_python_attachment() / analyze_spreadsheet(): run an attached .py / read a spreadsheet
- inspect_image() / transcribe_audio() / inspect_video(): inspect attached media
- youtube_research(url): find transcript/context for a YouTube URL

Tool call: {"type":"tool","tool":"web_search","args":{"query":"..."}}
File tools take empty args: {"type":"tool","tool":"attachment_summary","args":{}}
Final answer: {"type":"final","answer":"..."}

Rules:
- For facts about sources, articles, Wikipedia, dates, records, seasons, or "as of" info, do NOT answer from memory: web_search first, then visit_webpage the best result before finalizing.
- If a file is attached, inspect it with an attachment tool before finalizing.
- Never finalize by saying more research is needed; call another tool instead.
- Decode/transform the question if it is reversed or encoded, then answer what it actually asks.
- Match the requested answer format exactly. For "how many", answer with only the number. No extra words, labels, or lists unless requested.
"""


class GaiaAgent:
    def __init__(
        self,
        model_client: ModelClient | None = None,
        web_search: WebSearchTool | None = None,
        attachment_tool: AttachmentTool | None = None,
        multimodal_tool: MultimodalTool | None = None,
    ):
        self.model_client = model_client or ModelClient()
        self.web_search = web_search or WebSearchTool()
        self.attachment_tool = attachment_tool or AttachmentTool()
        self.multimodal_tool = multimodal_tool or MultimodalTool(
            attachment_tool=self.attachment_tool,
            web_search=self.web_search,
        )
        self.web_enabled = os.getenv("WEB_SEARCH_ENABLED", "true").lower() == "true"
        self.max_steps = int(os.getenv("AGENT_MAX_STEPS", "5"))

    def run(
        self,
        question: str,
        task_id: str,
        file_name: str | None = None,
    ) -> str:
        """
        Solve one GAIA task and return only the final answer.
        """
        history = [
            {
                "role": "user",
                "content": self._build_initial_message(question, task_id, file_name),
            }
        ]
        attachment_context = (
            AttachmentContext(task_id=task_id, file_name=file_name)
            if file_name
            else None
        )

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

            if action.get("type") == "tool":
                tool_result = self._execute_tool(action, attachment_context)
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

    def _build_initial_message(
        self,
        question: str,
        task_id: str,
        file_name: str | None,
    ) -> str:
        message = f"Question:\n{question}"
        if file_name:
            message += (
                f"\n\nAttached file available:\n"
                f"Task ID: {task_id}\n"
                f"File name: {file_name}\n"
                "Use attachment tools to inspect it before answering."
            )
        return message

    def _execute_tool(
        self,
        action: dict,
        attachment_context: AttachmentContext | None = None,
    ) -> str:
        tool = action.get("tool")
        args = action.get("args") or {}
        try:
            if tool == "web_search":
                if not self.web_enabled:
                    return "Tool error: web search is disabled."
                query = str(args.get("query", "")).strip()
                if not query:
                    return "Tool error: missing query."
                results = self.web_search.search(query)
                return self._format_search_results(results)

            if tool == "visit_webpage":
                if not self.web_enabled:
                    return "Tool error: web browsing is disabled."
                url = str(args.get("url", "")).strip()
                if not url:
                    return "Tool error: missing url."
                return self.web_search.read_page(url)

            if tool == "attachment_summary":
                if attachment_context is None:
                    return "Tool error: no attachment is available for this task."
                return self.attachment_tool.summary(attachment_context)

            if tool == "read_attachment_text":
                if attachment_context is None:
                    return "Tool error: no attachment is available for this task."
                return self.attachment_tool.read_text(attachment_context)

            if tool == "execute_python_attachment":
                if attachment_context is None:
                    return "Tool error: no attachment is available for this task."
                return self.attachment_tool.execute_python(attachment_context)

            if tool == "analyze_spreadsheet":
                if attachment_context is None:
                    return "Tool error: no attachment is available for this task."
                return self.attachment_tool.analyze_spreadsheet(attachment_context)

            if tool == "inspect_image":
                if attachment_context is None:
                    return "Tool error: no attachment is available for this task."
                return self.multimodal_tool.inspect_image(attachment_context)

            if tool == "transcribe_audio":
                if attachment_context is None:
                    return "Tool error: no attachment is available for this task."
                return self.multimodal_tool.transcribe_audio(attachment_context)

            if tool == "inspect_video":
                if attachment_context is None:
                    return "Tool error: no attachment is available for this task."
                return self.multimodal_tool.inspect_video(attachment_context)

            if tool == "youtube_research":
                if not self.web_enabled:
                    return "Tool error: web search is disabled."
                url = str(args.get("url", "")).strip()
                return self.multimodal_tool.youtube_research(url)

            return f"Tool error: unknown tool {tool!r}."
        except Exception as error:
            return f"Tool error: {error}"

    def _format_history(self, history: list[dict]) -> str:
        """
        Build the prompt history while keeping token use bounded: always keep the
        first message (the question), then add the most recent messages within a
        character budget, truncating any single long tool result. This stops web
        page text from compounding across steps and blowing the daily token cap.
        """
        per_item_limit = int(os.getenv("HISTORY_ITEM_CHAR_LIMIT", "2500"))
        total_limit = int(os.getenv("HISTORY_TOTAL_CHAR_LIMIT", "9000"))

        def render(item: dict) -> str:
            content = str(item["content"])
            if len(content) > per_item_limit:
                content = content[:per_item_limit] + "\n...[truncated]"
            return f"{item['role'].upper()}:\n{content}"

        if not history:
            return ""
        first = render(history[0])
        recent: list[str] = []
        used = len(first)
        for item in reversed(history[1:]):
            rendered = render(item)
            if used + len(rendered) > total_limit:
                break
            recent.append(rendered)
            used += len(rendered)
        recent.reverse()
        return "\n\n".join([first, *recent])

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
                "using the conversation and tool results. Return only the answer "
                "text with no explanation, no labels, and no JSON. Do not return a "
                "tool call. If the evidence is insufficient, give your single best "
                "guess as a plain answer."
            ),
            user_prompt=f"Original question:\n{question}\n\nHistory:\n{self._format_history(history)}",
        )
        cleaned = self._clean_answer_for_question(answer, question)
        if self._looks_like_tool_json(cleaned):
            cleaned = ""
        fallback = self._clean_answer(last_response)
        if self._looks_like_tool_json(fallback):
            fallback = ""
        return cleaned or fallback

    def _looks_like_tool_json(self, text: str) -> bool:
        """Detect leaked protocol JSON so it is never returned as a final answer."""
        stripped = text.strip()
        if not stripped.startswith("{"):
            return False
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return '"type"' in stripped and '"tool"' in stripped
        return isinstance(parsed, dict) and parsed.get("type") in {"tool", "final"}

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
        cleaned = cleaned.strip().strip('"').strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.startswith("text"):
                cleaned = cleaned[4:].strip()
        return cleaned.strip()
