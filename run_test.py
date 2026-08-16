from pathlib import Path
from render import PageRenderer

def test_renderer():
    html_file = Path("temp_render_test/test.html")
    output_dir = Path("temp_render_test/output")
    
    with PageRenderer(want_screenshot=True, want_ax_tree=True) as renderer:
        result = renderer.render(
            html_path=html_file,
            out_dir=output_dir,
            web_url_id="test_001"
        )
    
    print(f"Screenshot path: {result.screenshot_path}")
    print(f"AX Tree keys: {list(result.ax_tree.keys()) if result.ax_tree else 'None'}")

if __name__ == "__main__":
    test_renderer()
