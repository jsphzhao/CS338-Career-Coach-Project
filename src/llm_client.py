import logging
import os
from typing import Iterable, Mapping

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


LOGGER = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            if OpenAI is None:  # pragma: no cover
                raise ImportError(
                    "openai package is not installed. Please add it to requirements."
                )
            self._client = OpenAI(api_key=self.api_key)
        else:
            self._client = None

    def is_configured(self) -> bool:
        return self._client is not None

    def generate(
        self,
        messages: Iterable[Mapping[str, str]],
        temperature: float = 0.2,
        max_output_tokens: int | None = None,
    ) -> str:
        if not self._client:
            raise RuntimeError(
                "LLM client is not configured. Set OPENAI_API_KEY before running the server."
            )

        try:
            response = self._client.responses.create(
                model=self.model,
                input=list(messages),
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            output_text = getattr(response, "output_text", None)
            if output_text is None:  # pragma: no cover - backward compatibility
                output_text = response.output[0].content[0].text  # type: ignore[attr-defined]
            return output_text
        except AttributeError:
            # Fallback for older openai client versions with chat.completions
            chat_messages = list(messages)
            response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )
            return response.choices[0].message.content  # type: ignore[index]

