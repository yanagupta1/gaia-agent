import os
import json
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ModelConfig:
    provider: str = os.getenv("MODEL_PROVIDER", "ollama")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2:7b")
    timeout_seconds: int = int(os.getenv("MODEL_TIMEOUT_SECONDS", "120"))


class ModelClient:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.config.provider == "ollama":
            return self._generate_ollama(system_prompt, user_prompt)

        raise ValueError(
            f"Unsupported MODEL_PROVIDER={self.config.provider!r}. "
            "V0.1 supports MODEL_PROVIDER=ollama."
        )

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.config.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.config.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": 0,
                "num_predict": 128,
            },
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed: HTTP {error.code}: {detail}") from error
        return data.get("message", {}).get("content", "").strip()
