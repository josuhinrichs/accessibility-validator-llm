import base64
from pathlib import Path
from typing import List
from openai import OpenAI

from pathlib import Path


def image_path_to_data_url(image_path: str) -> str:
    """
    Converte uma imagem local em uma data URL para envio ao OpenAI.
    """
    suffix = Path(image_path).suffix.lower()
    mime_type = "image/png" if suffix == ".png" else "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


class LLMClient:
    def __init__(self,
                 models: List[str],
                 api_key: str | None = None,
                 base_url: str | None = None,
                 temperature: float = 0.0,
                 max_tokens: int = 16384,
                 force_json: bool = False,
                 include_images: bool = False,
                 ):
        self.models = models
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.force_json = force_json
        self.include_images = include_images

        client_kwargs = {}
        client_kwargs["api_key"] = self.api_key or "lm-studio"
        client_kwargs["base_url"] = self.base_url or "http://localhost:1234/v1"
        self.client = OpenAI(**client_kwargs)

    def run(self, model: str, prompt: str, images: List[str]) -> str:
        content = [{"type": "text", "text": prompt}]

        if self.include_images:
            for image_path in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_path_to_data_url(image_path)},
                    }
                )

        # if self.force_json:
        #     kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response.choices[0].message.content or "_Empty response"
