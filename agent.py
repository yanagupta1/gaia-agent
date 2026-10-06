from model import ModelClient


SYSTEM_PROMPT = """You are a GAIA benchmark answer agent.
Answer the user's question using only your reasoning and prior knowledge.
These tasks often include simple transformations, reversed text, tables, wordplay, and strict output formats.
Decode or transform the question when needed, reason silently, and then answer.
Return only the final answer, with no explanation, no markdown, and no label.
If the question requests a specific format, follow it exactly.
"""


class GaiaAgent:
    def __init__(self, model_client: ModelClient | None = None):
        self.model_client = model_client or ModelClient()

    def run(
        self,
        question: str,
        task_id: str,
        file_name: str | None = None,
    ) -> str:
        """
        Solve one GAIA task and return only the final answer.
        """
        prompt = self._build_prompt(question=question, task_id=task_id, file_name=file_name)
        answer = self.model_client.generate(SYSTEM_PROMPT, prompt)
        return self._clean_answer(answer)

    def _build_prompt(self, question: str, task_id: str, file_name: str | None) -> str:
        file_note = (
            f"\nAttached file name: {file_name}\n"
            "V0.1 cannot inspect attached files. If the answer requires the file, still give your best exact answer."
            if file_name
            else ""
        )
        return f"""Task ID: {task_id}
Question:
{question}
{file_note}

Important:
- Reason silently before answering.
- If the question text appears reversed, read it in reverse first.
- Do not include reasoning, labels, or commentary.

Return only the exact answer."""

    def _clean_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        prefixes = ("FINAL ANSWER:", "Final answer:", "Answer:", "ANSWER:")
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned.strip().strip('"').strip()
