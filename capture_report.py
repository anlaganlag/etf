import asyncio
from playwright.async_api import async_playwright
import sys

async def capture_screenshot(url, output_path):
    async with async_playwright() as p:
        # Using chromium
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        print(f"Navigating to {url}...")
        try:
            # Wait for the page to load
            await page.goto(url, wait_until="networkidle", timeout=60000)
            
            # Wait a bit more for charts to animate/render
            print("Waiting for charts to render...")
            await asyncio.sleep(5) 
            
            # Take a full page screenshot
            print(f"Capturing screenshot to {output_path}...")
            await page.screenshot(path=output_path, full_page=True)
            print("Done!")
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    url = "http://localhost:10110/#/performance/9d88cf8d-fcd0-11f0-bea9-00ffda9d6e63/reportView?token=e2e96d719135bbf887df4ee5c633b97d9e1aa482&orgCode=yingda&reportName=etf%E6%BB%9A%E5%8A%A8&startTime=2021-12-02T08:00:00Z&endTime=2026-01-23T08:00:00Z&benchmark=SHSE.000300"
    output = "strategy_report.png"
    asyncio.run(capture_screenshot(url, output))
