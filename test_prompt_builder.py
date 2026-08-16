import unittest
from prompt_builder import PromptRecipe, RenderedPage

class TestPromptBuilder(unittest.TestCase):
    def test_build_user_prompt(self):
        page = RenderedPage(web_url_id="1", html="<html></html>")
        prompt = PromptRecipe.build_user_prompt(page, evidence_inputs=["html"])
        
        self.assertIn("Audit this page", prompt)
        self.assertIn("DOM (HTML):", prompt)

if __name__ == "__main__":
    unittest.main()
