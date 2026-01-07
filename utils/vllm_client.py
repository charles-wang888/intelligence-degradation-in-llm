"""
vLLM API client using the OpenAI-compatible endpoints.
"""

import json
import logging
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)


class VLLMClient:
    """Minimal client for a vLLM OpenAI-compatible server."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: Optional[str] = None, timeout: int = 300):
        """
        Args:
            base_url: Base URL of the vLLM server. Either with or without the trailing /v1.
            api_key: API key if the server enforces authentication.
            timeout: Request timeout in seconds.
        """
        base = base_url.rstrip("/")
        if not base.lower().endswith("/v1"):
            base = f"{base}/v1"
        self.api_base = base
        self.chat_url = f"{self.api_base}/chat/completions"
        self.models_url = f"{self.api_base}/models"
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> Dict:
        """
        Generate text with the given model.

        Args:
            model: Model name served by vLLM.
            prompt: Input prompt as a plain string.
            temperature: Sampling temperature.
            max_tokens: Max new tokens to generate.
            stream: Whether to stream tokens back.

        Returns:
            Dict containing the generated response text and raw response payload.
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        try:
            response = requests.post(
                self.chat_url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
                stream=stream,
            )
            response.raise_for_status()

            if stream:
                full_response = ""
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    # vLLM streams lines prefixed with "data:"
                    if line.startswith("data:"):
                        line = line[len("data:") :].strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices", [])
                    for choice in choices:
                        delta = choice.get("delta", {}) or choice.get("message", {})
                        text_chunk = delta.get("content") or delta.get("text")
                        if text_chunk:
                            full_response += text_chunk
                return {"response": full_response, "done": True}

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return {"response": "", "raw": data}
            message = choices[0].get("message", {}) or choices[0].get("delta", {})
            content = message.get("content") or ""
            return {"response": content, "raw": data}

        except requests.exceptions.Timeout:
            logger.error(f"vLLM request timed out: model={model}, timeout={self.timeout}s")
            return {"error": "timeout", "response": ""}
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM request failed: {e}")
            return {"error": str(e), "response": ""}
        except Exception as e:
            logger.error(f"Unexpected error when calling vLLM: {e}")
            return {"error": str(e), "response": ""}

    def check_model_available(self, model: str) -> bool:
        """
        Check whether the model is served by the vLLM instance.
        """
        try:
            response = requests.get(
                self.models_url,
                headers=self._headers(),
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("data") or data.get("models") or []
            for item in models:
                name = item.get("id") or item.get("name")
                if name == model:
                    return True
            logger.error(f"Model {model} not found on vLLM server")
            return False
        except Exception as e:
            logger.error(f"Failed to list models from vLLM: {e}")
            return False
