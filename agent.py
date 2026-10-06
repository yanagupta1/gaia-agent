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
        answer = self.model_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=question,
        )
        return self._clean_answer(answer)

    def _clean_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        prefixes = ("FINAL ANSWER:", "Final answer:", "Answer:", "ANSWER:")
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :].strip()
        return cleaned.strip().strip('"').strip()
