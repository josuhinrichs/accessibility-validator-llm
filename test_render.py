import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from render import PageRenderer

class TestPageRenderer(unittest.TestCase):
    @patch("render.sync_playwright")
    def test_renderer_lifecycle(self, mock_playwright):
        # Mock setup
        mock_p = MagicMock()
        mock_playwright.return_value.start.return_value = mock_p
        
        with PageRenderer(want_screenshot=False, want_ax_tree=False) as renderer:
            # Check if browser was launched
            self.assertIsNotNone(renderer._browser)
        
        # Check if cleanup happened
        mock_p.chromium.launch.return_value.close.assert_called()

if __name__ == "__main__":
    unittest.main()
