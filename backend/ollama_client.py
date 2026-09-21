"""
ollama_client.py
----------------
Thin wrapper around the local Ollama HTTP server (default http://localhost:11434).

Ollama runs the LLM entirely on your machine, so there is no API key, no cloud
call and no cost.  Every failure is converted into a clear, human-readable
message that the frontend can display.
"""

from typing import List, Optional

import requests

from backend import config
from backend.logger import get_logger, shorten

log = get_logger("cognify.ollama")


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable, the model is missing or generation fails."""


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

    # ------------------------------------------------------------------ #
    # Health helpers
    # ------------------------------------------------------------------ #
    def list_models(self) -> List[str]:
        """Names of the models installed locally. Empty list if Ollama is down."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            return [m.get("name", "") for m in response.json().get("models", [])]
        except Exception as exc:  # noqa: BLE001 - we intentionally swallow everything
            log.warning("Could not list Ollama models: %s", exc)
            return []

    def is_running(self) -> bool:
        try:
            requests.get(f"{self.host}/api/tags", timeout=5).raise_for_status()
            return True
        except Exception:  # noqa: BLE001
            return False

    def health(self) -> dict:
        """Structured status used by GET /api/health."""
        models = self.list_models()
        running = bool(models) or self.is_running()

        # "llama3.2:3b" should also match an installed "llama3.2:3b-instruct-q4_K_M"
        available = any(m == self.model or m.startswith(self.model.split(":")[0]) for m in models)

        if not running:
            message = (
                "Ollama is not running. Open a new terminal and run:  ollama serve   "
                "(on Windows, starting the Ollama app also starts the server)."
            )
        elif not available:
            message = f"Model '{self.model}' is not installed. Run:  ollama pull {self.model}"
        else:
            message = "Ollama is running and the configured model is available."

        return {
            "running": running,
            "model_available": available,
            "models": models,
            "message": message,
        }

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        num_predict: Optional[int] = None,
    ) -> str:
        """Send a prompt to Ollama and return the generated text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.OLLAMA_TEMPERATURE if temperature is None else temperature,
                "num_predict": config.OLLAMA_NUM_PREDICT if num_predict is None else num_predict,
            },
        }
        if system:
            payload["system"] = system

        log.debug("Ollama prompt: %s", shorten(prompt, 400))

        try:
            response = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}. Start it with 'ollama serve' "
                f"and make sure the Ollama app is installed."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaError(
                f"Ollama timed out after {self.timeout}s. Try a smaller model "
                f"(for example llama3.2:1b) or increase OLLAMA_TIMEOUT in .env."
            ) from exc

        if response.status_code == 404:
            raise OllamaError(
                f"Model '{self.model}' was not found in Ollama. "
                f"Install it with:  ollama pull {self.model}"
            )
        if response.status_code >= 400:
            raise OllamaError(f"Ollama returned HTTP {response.status_code}: {shorten(response.text)}")

        try:
            text = response.json().get("response", "")
        except ValueError as exc:
            raise OllamaError("Ollama returned a response that is not valid JSON.") from exc

        if not text.strip():
            raise OllamaError("Ollama returned an empty response. Try running the query again.")

        log.debug("Ollama answer: %s", shorten(text))
        return text.strip()
