from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, urlunparse

import requests

from .models import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        merged_system_prompt = self._merged_system_prompt(system_prompt)
        if self.config.provider == "ollama":
            return self._generate_ollama(prompt, merged_system_prompt)
        return self._generate_http(prompt, merged_system_prompt)

    def test_connection(self) -> dict[str, Any]:
        models = self.list_models()
        return {
            "status": "ok",
            "provider": self.config.provider,
            "message": "Connection successful.",
            "model_count": len(models),
            "models": models[:20],
        }

    def list_models(self) -> list[str]:
        if self.config.provider == "ollama":
            return self._list_ollama_models()
        return self._list_http_models()

    def _generate_ollama(self, prompt: str, system_prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        if system_prompt:
            payload["system"] = system_prompt

        body = self._ollama_request("post", "/api/generate", json=payload)
        if not isinstance(body, dict):
            raise ValueError("Unexpected Ollama response format.")

        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Ollama response does not contain usable text.")
        return text.strip()

    def _generate_http(self, prompt: str, system_prompt: str) -> str:
        endpoint = self.config.endpoint or self.config.base_url
        headers = self._headers()

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system_prompt,
            "temperature": self.config.temperature,
        }

        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        text = self._extract_text(body)
        if not text:
            raise ValueError("Unable to extract text from the HTTP LLM response.")
        return text

    def _list_ollama_models(self) -> list[str]:
        body = self._ollama_request("get", "/api/tags")
        models = self._extract_model_names(body)
        return models

    def _ollama_request(self, method: str, path: str, **kwargs: Any) -> Any:
        errors: list[str] = []
        for base_url in self._ollama_base_candidates():
            url = f"{base_url}{path}"
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    timeout=self.config.timeout_seconds,
                    **kwargs,
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                errors.append(f"{url}: {exc}")

        details = " | ".join(errors[:3]) if errors else "no URL candidate available"
        raise RuntimeError(
            "Unable to reach Ollama. Ensure it is running and accessible from the backend. "
            f"Tried: {details}"
        )

    def _ollama_base_candidates(self) -> list[str]:
        primary = self._normalize_base_url(self.config.base_url)
        parsed = urlparse(primary)

        candidates: list[str] = [primary]
        host = (parsed.hostname or "").lower()
        port = parsed.port or 11434
        scheme = parsed.scheme or "http"

        if host in {"localhost", "127.0.0.1", "::1"}:
            for alt_host in ("localhost", "127.0.0.1", "host.docker.internal"):
                candidates.append(
                    urlunparse((scheme, f"{alt_host}:{port}", "", "", "", "")).rstrip("/")
                )

        # Deduplicate while preserving order
        unique: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            clean = item.rstrip("/")
            if clean and clean not in seen:
                seen.add(clean)
                unique.append(clean)
        return unique

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        value = base_url.strip()
        if not value:
            return "http://localhost:11434"
        if "://" not in value:
            value = f"http://{value}"
        return value.rstrip("/")

    def _list_http_models(self) -> list[str]:
        headers = self._headers()
        urls = self._http_model_urls()
        errors: list[str] = []

        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=self.config.timeout_seconds)
                response.raise_for_status()
                body = response.json()
                models = self._extract_model_names(body)
                if models:
                    return models
                errors.append(f"{url}: response did not contain models")
            except requests.RequestException as exc:
                errors.append(f"{url}: {exc}")

        joined = " | ".join(errors) if errors else "no model endpoint configured"
        raise RuntimeError(f"Unable to fetch model list from HTTP provider: {joined}")

    def _http_model_urls(self) -> list[str]:
        candidates: list[str] = []
        if self.config.endpoint:
            candidates.append(self._to_models_endpoint(self.config.endpoint))
        if self.config.base_url:
            candidates.append(self._to_models_endpoint(self.config.base_url))

        # Deduplicate while keeping order
        seen: set[str] = set()
        urls: list[str] = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                urls.append(candidate)
        return urls

    @staticmethod
    def _to_models_endpoint(url: str) -> str:
        cleaned = url.rstrip("/")
        lowered = cleaned.lower()
        suffixes = ("/chat/completions", "/completions", "/responses", "/generate")
        for suffix in suffixes:
            if lowered.endswith(suffix):
                return f"{cleaned[: -len(suffix)]}/models"
        if lowered.endswith("/models"):
            return cleaned
        return f"{cleaned}/models"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.headers}
        if self.config.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _merged_system_prompt(self, call_system_prompt: str) -> str:
        global_system_prompt = (self.config.system_prompt or "").strip()
        local_system_prompt = (call_system_prompt or "").strip()
        if global_system_prompt and local_system_prompt:
            return f"{global_system_prompt}\n\n{local_system_prompt}"
        if local_system_prompt:
            return local_system_prompt
        return global_system_prompt

    @staticmethod
    def _extract_text(body: Any) -> str:
        if isinstance(body, str):
            return body.strip()
        if isinstance(body, dict):
            for key in ("response", "text", "output", "content"):
                value = body.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            choices = body.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
                    content = first.get("text")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        return ""

    @staticmethod
    def _extract_model_names(body: Any) -> list[str]:
        values: list[str] = []

        def collect(item: Any) -> None:
            if isinstance(item, str):
                values.append(item.strip())
                return
            if isinstance(item, dict):
                for key in ("name", "id", "model"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        values.append(value.strip())
                        return
            if isinstance(item, list):
                for element in item:
                    collect(element)

        if isinstance(body, dict):
            for key in ("models", "data", "results"):
                if key in body:
                    collect(body[key])
        elif isinstance(body, list):
            collect(body)

        # Deduplicate and sort for stable dropdown display
        return sorted({value for value in values if value})
