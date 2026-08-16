import logging
from llm_clients import LLMClient
from render import PageRenderer, RenderedPage
from prompt_builder import PromptRecipe
from pathlib import Path
import json

# Setup minimal logging
logging.basicConfig(level=logging.INFO)

def test_full_pipeline_with_ollama():
    # 1. Initialize Renderer
    renderer = PageRenderer(want_screenshot=True, want_ax_tree=True)
    
    # Use a dummy HTML file (ensure it exists)
    html_path = Path("temp_render_test/test.html")
    out_dir = Path("temp_render_test/output")
    web_url_id = "test_ollama_001"
    
    # 2. Initialize Ollama Client (Qwen2.5-VL)
    client = LLMClient(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        models=["qwen2.5vl"],
        include_images=True
    )
    
    # 3. Render
    print("--- Starting Render ---")
    with renderer:
        rendered_page = renderer.render(
            html_path=html_path,
            out_dir=out_dir,
            web_url_id=web_url_id
        )
    
    print(f"Rendered: screenshot={bool(rendered_page.screenshot_path)}, ax_tree={bool(rendered_page.ax_tree)}")
    
    # 4. Build Prompt
    print("--- Building Prompt ---")
    evidence_inputs = {"html", "axtree", "screenshot"}
    prompt = PromptRecipe.build_user_prompt(
        page=rendered_page,
        evidence_inputs=evidence_inputs,
        taxonomy_block="WCAG 2.1 Level AA"
    )
    
    # 5. Run LLM
    print("--- Running LLM (Qwen2.5-VL) ---")
    images = [rendered_page.screenshot_path] if rendered_page.screenshot_path else []
    
    try:
        response = client.run(
            model="qwen2.5vl",
            prompt=prompt,
            images=images
        )
        print("\n--- LLM Output ---")
        print(response)
    except Exception as e:
        print(f"\n--- LLM Error ---")
        print(e)

if __name__ == "__main__":
    test_full_pipeline_with_ollama()
