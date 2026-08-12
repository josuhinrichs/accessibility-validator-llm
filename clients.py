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


class OpenAIInferenceClient:
    """
    Pequeno wrapper para manter a interface do pipeline parecida com a do Ollama,
    mas executando chamadas para a OpenAI.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        client_kwargs = {}

        resolved_base_url = base_url or "http://10.102.20.26:1234/v1"
        resolved_api_key = api_key or "lm-studio"

        client_kwargs["api_key"] = resolved_api_key
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url

        self.client = OpenAI(**client_kwargs)

    def generate(self, model: str, prompt: str, images: List[str]) -> str:
        content = [{"type": "text", "text": prompt}]

        for image_path in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_to_data_url(image_path)},
                }
            )

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
        )

        return response.choices[0].message.content or ""


openai_client = OpenAIInferenceClient(
    api_key="lm-studio",
    base_url="http://10.102.20.26:1234/v1",
)

ollama_client = OpenAIInferenceClient(
    api_key="ollama",
    base_url="http://localhost:1234/v1",
)