from openai import OpenAI

client = OpenAI(base_url="http://10.102.20.26:1234/v1", api_key="lm-studio")

modelo = "google/gemma-4-31b"

response = client.chat.completions.create(
    messages = [{"role":"system", "content":"You are a helpful assistant"},
                {"role":"user", "content":"Hello, say each letter from the text 'Calculus' separately by line breaks"}],
    model = modelo)


print(response.choices[0].message.content)
