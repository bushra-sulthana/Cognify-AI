
"""
Thin wrapper around the local Ollama HTTP server.
"""

from typing import List, Optional

import re

import requests

from backend import config
from backend.logger import get_logger, shorten

log = get_logger("cognify.ollama")


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or generation fails."""


def _is_thinking_model(name: str) -> bool:
    return any(key in name for key in ("qwen3", "deepseek-r1", "qwq", "magistral"))


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> reasoning that some models print before the answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    if "</think>" in text:                         # opening tag missing: drop everything before it
        text = text.split("</think>", 1)[1]
    if "<think>" in text.lower():                  # reasoning never finished (cut off)
        text = text[: text.lower().index("<think>")]
    return text.strip()


class OllamaClient:
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.OLLAMA_TIMEOUT

    def list_models(self) -> List[str]:
        try:
            response = requests.get(
                f"{self.host}/api/tags", timeout=5
            )
            response.raise_for_status()
            return [
                model.get("name", "")
                for model in response.json().get("models", [])
            ]
        except Exception as exc:
            log.warning("Could not list Ollama models: %s", exc)
            return []

    def is_running(self) -> bool:
        try:
            response = requests.get(
                f"{self.host}/api/tags", timeout=5
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    def health(self) -> dict:
        models = self.list_models()
        running = bool(models) or self.is_running()

        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        available = any(name == self.model or name == wanted for name in models)

        if not running:
            message = (
                "Ollama is not running. Start the Ollama application "
                "or run ollama serve."
            )
        elif not available:
            message = (
                f"Model '{self.model}' is not installed. "
                f"Run: ollama pull {self.model}"
            )
        else:
            message = "Ollama is running and the configured model is available."

        return {
            "running": running,
            "model_available": available,
            "models": models,
            "message": message,
        }

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
        model: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {
                "num_ctx": config.OLLAMA_NUM_CTX,
                "temperature": (
                    config.OLLAMA_TEMPERATURE
                    if temperature is None
                    else temperature
                ),
                "num_predict": (
                    config.OLLAMA_NUM_PREDICT
                    if num_predict is None
                    else num_predict
                ),
            },
        }

        used_model = (model or self.model).lower()
        if _is_thinking_model(used_model):
            payload["think"] = False             # Ollama >= 0.9: turn the reasoning phase off
            prompt_note = "\n\n/no_think"        # qwen3 soft switch for older Ollama versions
            payload["prompt"] = prompt + prompt_note

        if system:
            payload["system"] = system
        if images:
            payload["images"] = images        # base64 strings, vision models only

        log.debug("Ollama prompt: %s", shorten(prompt, 400))

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}. "
                "Start the Ollama application."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaError(
                f"Ollama timed out after {self.timeout} seconds. "
                "Increase OLLAMA_TIMEOUT in your configuration."
            ) from exc

        if response.status_code == 404:
            raise OllamaError(
                f"Model '{self.model}' was not found. "
                f"Run: ollama pull {self.model}"
            )

        if response.status_code >= 400:
            raise OllamaError(
                f"Ollama returned HTTP {response.status_code}: "
                f"{shorten(response.text, 500)}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError(
                "Ollama returned a response that is not valid JSON."
            ) from exc

        text = data.get("response", "")

        if not isinstance(text, str) or not text.strip():
            log.error(
                "Empty Ollama answer. Model=%s, done=%s, "
                "reason=%s, keys=%s",
                self.model,
                data.get("done"),
                data.get("done_reason"),
                list(data.keys()),
            )
            raise OllamaError(
                "Ollama generated no answer text. Check backend logs."
            )

        text = strip_reasoning(text)
        if not text:
            raise OllamaError(
                "The model only produced its internal reasoning and no answer. "
                "Use a non-reasoning model (e.g. llama3.2:1b) or raise OLLAMA_NUM_PREDICT."
            )

        log.debug("Ollama answer: %s", shorten(text))
        return text
