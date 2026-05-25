"""Capture a screenshot of the Next.js UI showing real search results.

Assumes:
  - API server is running at http://127.0.0.1:8000 (cdrag serve)
  - Next.js dev/build server is running at http://localhost:3000

Output:
  docs/images/ui-search.png
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("docs/images/ui-search.png")
URL = "http://localhost:3000"
QUERY = "How do I stream messages from the Claude API?"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 1600},
            device_scale_factor=2,
            color_scheme="dark",
        )
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")

        page.fill("#q", QUERY)
        page.locator("button[type=submit]").click()

        page.locator("ul li").first.wait_for(state="visible", timeout=120_000)
        page.wait_for_timeout(800)

        page.screenshot(path=str(OUT), full_page=True)
        print(f"saved: {OUT.resolve()}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
