import os
import json
import re
import time
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
    groq_api_url: str = os.getenv(
        "GROQ_API_URL",
        "https://api.groq.com/openai/v1/chat/completions",
    )
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    groq_token: str = os.getenv("GROQ_API_KEY", "")
    timeout_seconds: int = int(os.getenv("MODEL_TIMEOUT_SECONDS", "120"))
    max_tokens: int = int(os.getenv("MODEL_MAX_TOKENS", "512"))
    max_retries: int = int(os.getenv("MODEL_MAX_RETRIES", "5"))
    max_retry_wait_seconds: float = float(os.getenv("MODEL_MAX_RETRY_WAIT_SECONDS", "45"))


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
            if not self.config.hf_token:
                raise RuntimeError(
                    "HF_TOKEN is required when MODEL_PROVIDER=hf. Add it as a "
                    "Hugging Face Space secret before running evaluation."
                )
            return self._generate_openai_compatible(
                system_prompt,
                user_prompt,
                api_url=self.config.hf_api_url,
                model=self.config.hf_model,
                token=self.config.hf_token,
                provider_label="Hugging Face router",
                json_mode=json_mode,
            )

        if self.config.provider == "groq":
            if not self.config.groq_token:
                raise RuntimeError(
                    "GROQ_API_KEY is required when MODEL_PROVIDER=groq. Get a free "
                    "key at console.groq.com and add it as a Space secret."
                )
            return self._generate_openai_compatible(
                system_prompt,
                user_prompt,
                api_url=self.config.groq_api_url,
                model=self.config.groq_model,
                token=self.config.groq_token,
                provider_label="Groq",
                json_mode=json_mode,
            )

        raise ValueError(
            f"Unsupported MODEL_PROVIDER={self.config.provider!r}. "
            "Supported providers: ollama, hf, groq."
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
        except TimeoutError as error:
            raise RuntimeError(
                f"Ollama request timed out after {self.config.timeout_seconds}s. "
                "Local 7B models can be slow; raise MODEL_TIMEOUT_SECONDS or use a "
                "smaller/faster model."
            ) from error
        except OSError as error:
            raise RuntimeError(
                "Ollama is not reachable. For local runs, start Ollama or set "
                "OLLAMA_BASE_URL. For Hugging Face Spaces, set MODEL_PROVIDER to a "
                "hosted provider before running evaluation."
            ) from error
        return data.get("message", {}).get("content", "").strip()

    def _generate_openai_compatible(
        self,
        system_prompt: str,
        user_prompt: str,
        api_url: str,
        model: str,
        token: str,
        provider_label: str,
        json_mode: bool = False,
    ) -> str:
        payload = {
            "model": model,
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
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "gaia-agent/0.5",
                "Accept": "application/json",
            },
            method="POST",
        )
        data = self._send_with_retry(request, provider_label)
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError(f"{provider_label} returned no choices: {data}")
        return choices[0].get("message", {}).get("content", "").strip()

    def _send_with_retry(self, request, provider_label: str) -> dict:
        """
        POST with retry/backoff on HTTP 429 (rate limit). Free Groq tiers have a
        low tokens-per-minute cap, so we honor the server's retry hint and wait.
        """
        last_detail = ""
        for attempt in range(self.config.max_retries + 1):
            try:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                last_detail = detail
                if error.code == 429 and attempt < self.config.max_retries:
                    wait = _parse_retry_after(error.headers, detail)
                    # A huge wait means a daily/hard cap, not a transient per-minute
                    # spike. Waiting it out would hang the run (and time out the
                    # Space), so fail fast and let the caller move on.
                    if wait > self.config.max_retry_wait_seconds:
                        raise RuntimeError(
                            f"{provider_label} rate limit exceeded and retry wait "
                            f"({wait:.0f}s) is too long (likely a daily token cap): "
                            f"{detail}"
                        ) from error
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"{provider_label} request failed: HTTP {error.code}: {detail}"
                ) from error
            except TimeoutError as error:
                self._raise_timeout(provider_label, error)
            except OSError as error:
                self._raise_unreachable(provider_label, error)
        raise RuntimeError(
            f"{provider_label} request failed after retries: {last_detail}"
        )

    def _raise_timeout(self, provider_label: str, error: Exception):
        raise RuntimeError(
            f"{provider_label} request timed out after "
            f"{self.config.timeout_seconds}s. Raise MODEL_TIMEOUT_SECONDS or "
            "check provider availability."
        ) from error

    def _raise_unreachable(self, provider_label: str, error: Exception):
        raise RuntimeError(
            f"{provider_label} is not reachable. Check Space networking, "
            "the API key, the model name, and provider availability."
        ) from error


def _parse_retry_after(headers, detail: str, default: float = 5.0) -> float:
    """Determine how long to wait before retrying a 429 response."""
    retry_after = headers.get("Retry-After") if headers else None
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    # Groq embeds a hint like "Please try again in 5.955s" in the body.
    match = re.search(r"try again in ([0-9.]+)s", detail)
    if match:
        try:
            return float(match.group(1)) + 0.5
        except ValueError:
            pass
    return default
