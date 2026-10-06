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
    hf_api_url: str = os.getenv(
        "HF_ROUTER_API_URL",
        "https://router.huggingface.co/v1/chat/completions",
    )
    hf_model: str = os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    hf_token: str = os.getenv("HF_TOKEN", "")
    timeout_seconds: int = int(os.getenv("MODEL_TIMEOUT_SECONDS", "30"))
    max_tokens: int = int(os.getenv("MODEL_MAX_TOKENS", "512"))


class ModelClient:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        if self.config.provider == "ollama":
            return self._generate_ollama(system_prompt, user_prompt, json_mode=json_mode)

        if self.config.provider in {"hf", "huggingface"}:
            return self._generate_huggingface(
                system_prompt,
                user_prompt,
                json_mode=json_mode,
            )

        raise ValueError(
            f"Unsupported MODEL_PROVIDER={self.config.provider!r}. "
            "Supported providers: ollama, hf."
        )

    def _generate_ollama(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
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
                "num_predict": self.config.max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
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
        except OSError as error:
            raise RuntimeError(
                "Ollama is not reachable. For local runs, start Ollama or set "
                "OLLAMA_BASE_URL. For Hugging Face Spaces, set MODEL_PROVIDER to a "
                "hosted provider before running evaluation."
            ) from error
        return data.get("message", {}).get("content", "").strip()

    def _generate_huggingface(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        if not self.config.hf_token:
            raise RuntimeError(
                "HF_TOKEN is required when MODEL_PROVIDER=hf. Add it as a "
                "Hugging Face Space secret before running evaluation."
            )

        payload = {
            "model": self.config.hf_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        request = Request(
            self.config.hf_api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.hf_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Hugging Face router request failed: HTTP {error.code}: {detail}"
            ) from error
        except OSError as error:
            raise RuntimeError(
                "Hugging Face router is not reachable. Check Space networking, "
                "HF_TOKEN, HF_MODEL, and provider availability."
            ) from error

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"Hugging Face router returned no choices: {data}")
        return choices[0].get("message", {}).get("content", "").strip()
