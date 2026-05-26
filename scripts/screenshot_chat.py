"""Capture a screenshot of the /chat page mid-streaming.

Assumes both servers are running:
  - cdrag serve --port 8000
  - npm run dev (in web/)

Output:
  docs/images/ui-chat.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("docs/images/ui-chat.png")
URL = "http://localhost:3000/chat"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 1400},
            device_scale_factor=2,
            color_scheme="dark",
        )
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")

        # Click a sample question that reliably retrieves streaming.md and
        # produces citations (per baseline.json by_id row 001).
        page.get_by_role(
            "button", name="How do I stream messages from the Claude API in Python?"
        ).click()

        # Wait for streaming to finish: the input area swaps the "Stop" button
        # back to "Send" when busy=false. This is the cleanest done signal
        # because the header also contains the word "citations".
        page.get_by_role("button", name="Send").wait_for(state="visible", timeout=180_000)
        page.wait_for_timeout(600)

        page.screenshot(path=str(OUT), full_page=True)
        print(f"saved: {OUT.resolve()}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
