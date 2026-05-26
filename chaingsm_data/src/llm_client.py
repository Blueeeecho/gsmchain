"""OpenAI-compatible LLM client with JSON parsing and backoff."""

from __future__ import annotations

import json
import time

from openai import OpenAI


class FatalLLMError(RuntimeError):
    """Non-retryable API failure, such as auth or insufficient balance."""


class LLMJSONParseError(ValueError):
    """JSON parsing failed, but the raw model content is useful for debugging."""

    def __init__(self, message, raw_content):
        super().__init__(message)
        self.raw_content = raw_content


class LLMClient:
    def __init__(self, model, api_key, base_url=None, api_retries=3, thinking=None):
        self.model = model
        self.api_retries = api_retries
        self.thinking = thinking
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def generate_json(self, messages, temperature=0.7, max_tokens=2000):
        content = self.generate_text(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return self._parse_json_content(content)

    def generate_text(self, messages, temperature=0.7, max_tokens=2000, response_format=None):
        last_error = None
        for attempt in range(1, self.api_retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if self.thinking in {"enabled", "disabled"}:
                    kwargs["extra_body"] = {"thinking": {"type": self.thinking}}
                if response_format is not None:
                    kwargs["response_format"] = response_format
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("empty response content")
                return content
            except Exception as exc:
                if self._is_fatal_error(exc):
                    raise FatalLLMError(str(exc)) from exc
                last_error = exc
                if attempt >= self.api_retries:
                    break
                time.sleep(2 ** (attempt - 1))
        raise RuntimeError(f"LLM request failed after {self.api_retries} attempts: {last_error}")

    @staticmethod
    def _parse_json_content(content):
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError as exc:
                    raise LLMJSONParseError(str(exc), content) from exc
            raise LLMJSONParseError("No JSON object found in response", content)

    @staticmethod
    def _is_fatal_error(exc):
        status_code = getattr(exc, "status_code", None)
        if status_code in {401, 402}:
            return True
        text = str(exc).lower()
        fatal_markers = [
            "insufficient balance",
            "authentication fails",
            "incorrect api key",
            "invalid api key",
        ]
        return any(marker in text for marker in fatal_markers)
