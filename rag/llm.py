# rag/llm.py
import json
from functools import lru_cache
from typing import TypeVar

from loguru import logger
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import ai_settings

T = TypeVar("T", bound=BaseModel)

# Errores pasajeros: reintentar la misma petición puede funcionar
TRANSIENT = (APITimeoutError, RateLimitError, APIConnectionError, InternalServerError)


class LLMOutputError(Exception):
    """La API respondió, pero el contenido no sirve (vacío, cortado o fuera de esquema).
    Guarda la salida cruda para que quien llama pueda pedir una corrección."""

    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


class TruncatedError(LLMOutputError):
    """La respuesta se cortó por max_tokens: pedir una corrección no sirve, hace falta más espacio."""


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(
        api_key=ai_settings.deepseek_api.get_secret_value(),
        base_url=ai_settings.llm_base_url,
        timeout=60,
        max_retries=0,
    )


def _log_retry(state) -> None:
    logger.warning(
        "Reintento #{} de la llamada al LLM por {!r}",
        state.attempt_number,
        state.outcome.exception(),
    )


@retry(
    retry=retry_if_exception_type(TRANSIENT),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    before_sleep=_log_retry,
    reraise=True,
)
def _chat(messages, model, temperature, max_tokens, thinking: bool):
    return get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
    )


def llm_complete(
    system: str,
    user: str,
    schema: type[T],
    *,
    temperature: float = 0.3,
    model: str | None = None,
    max_tokens: int = 2000,
    followup: list[dict] | None = None,
    thinking: bool = False,          # ← nuevo
) -> T:
    """Llama al LLM y devuelve la respuesta validada contra `schema`.

    - Errores de transporte: se reintentan aquí (en _chat).
    - Errores de contenido: se lanzan como LLMOutputError con la salida cruda;
      corregirlos es trabajo de quien llama (por ejemplo, generate.py).
    - `followup`: mensajes extra (respuesta anterior + corrección) para reparar.
    """
    model = model or ai_settings.llm_model
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    messages = [
        {
            "role": "system",
            "content": f"{system}\n\nResponde SOLO con un objeto JSON válido "
                       f"que cumpla este JSON Schema:\n{schema_json}",
        },
        {"role": "user", "content": user},
        *(followup or []),
    ]

    r = _chat(messages, model, temperature, max_tokens, thinking)
    choice = r.choices[0]
    if r.usage:
        logger.info("[llm] {}: {} in / {} out", model,
                    r.usage.prompt_tokens, r.usage.completion_tokens)

    raw = choice.message.content
    if choice.finish_reason == "length":
        raise TruncatedError(f"respuesta cortada por max_tokens={max_tokens}", raw)
    if not raw:
        raise LLMOutputError("respuesta vacía", raw)

    try:
        return schema.model_validate_json(raw)
    except ValidationError as e:
        raise LLMOutputError(f"la salida no cumple el esquema: {e}", raw) from e