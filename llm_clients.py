import base64
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from typing_extensions import final

from config import logger
from prompt_builder import SYSTEM_PROMPT_EVIDENCE_FIRST


def image_path_to_data_url(image_path: str) -> str:
    """
    Converte uma imagem local em uma data URL para envio ao OpenAI.
    """
    suffix = Path(image_path).suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"

@final
class LLMClient:
    def __init__(self,
                 models: list[str],
                 api_key: str | None = None,
                 base_url: str | None = None,
                 temperature: float = 0.0,
                 max_tokens: int = 16384,
                 force_json: bool = False,
                 include_images: bool = False,
                 system_prompt: str | None = SYSTEM_PROMPT_EVIDENCE_FIRST,
                 num_ctx: int | None = None,
                 ):
        self.models = models
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.force_json = force_json
        self.include_images = include_images
        self.system_prompt = system_prompt
        self.num_ctx = num_ctx

        client_kwargs = {}
        client_kwargs["api_key"] = self.api_key or "lm-studio"
        client_kwargs["base_url"] = self.base_url or "http://localhost:1234/v1"
        self.client = OpenAI(**client_kwargs)

    def run(self, model: str, prompt: str, images: list[str]) -> str:
        if self.include_images and images:
            content: str | list[dict[str, object]] = [{"type": "text", "text": prompt}]
            for image_path in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_path_to_data_url(image_path)},
                    }
                )
        else:
            content = prompt

        messages: list[dict[str, object]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})

        base_kwargs: dict[str, object] = {
            "model": model,
            "messages": cast(Any, messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        optional_kwargs: dict[str, object] = {}
        if self.num_ctx is not None:
            optional_kwargs["extra_body"] = {
                "options": {
                    "num_ctx": self.num_ctx,
                }
            }
        if self.force_json:
            optional_kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(
                **cast(Any, base_kwargs),
                **cast(Any, optional_kwargs),
            )
        except Exception as exc:
            # Fallback for OpenAI-compatible servers that don't support some optional fields.
            if optional_kwargs:
                logger.warning(
                    "llm_optional_params_unsupported_fallback",
                    model=model,
                    base_url=self.base_url,
                    dropped_params=list(optional_kwargs.keys()),
                    error=str(exc),
                )
                response = self.client.chat.completions.create(**cast(Any, base_kwargs))
            else:
                raise

        return response.choices[0].message.content or "_Empty response"

lm_studio_client = LLMClient(
    api_key="lm-studio",
    base_url="http://10.102.20.26:1234/v1",
    models=["llama-4-scout-17b-16e-instruct", "gemma-3-27b-it", "gemma-3-12b-it", "deepseek-r1-distill-llama-70b"],
    include_images=True,
    force_json=True,
)

# Backward-compatible alias
openai_client = lm_studio_client

ollama_client = LLMClient(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
    models=["qwen2.5vl", "qwen2.5-coder:7b-instruct"],
    include_images=True,
    force_json=True,
    num_ctx=32768,
)
