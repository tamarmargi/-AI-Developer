from mcp.server.fastmcp import FastMCP
from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright, Page, Browser, Playwright

mcp = FastMCP("weather-Israel")

FORECAST_URL = "https://www.weather2day.co.il/forecast"

_playwright: Playwright | None = None
_browser: Browser | None = None
_page: Page | None = None


@mcp.tool()
async def open_weather_forecast_israel() -> str:
    """Opens a browser and navigates to the Israeli weather forecast website."""
    global _playwright, _browser, _page

    if _page is not None:
        return f"Browser is already open at {_page.url}"

    _playwright = await async_playwright().start()
    _browser = await _playwright.chromium.launch(headless=False)
    _page = await _browser.new_page()
    await _page.goto(FORECAST_URL, wait_until="domcontentloaded")

    return f"Opened browser and navigated to {FORECAST_URL}"


@mcp.tool()
async def enter_weather_forecast_city_israel(city: str) -> str:
    """
    Enters a city name into the search field on the weather forecast page.

    Args:
        city: The name of the city to search for (in Hebrew or English).
    """
    global _page

    if _page is None:
        return "Error: browser is not open. Call open_weather_forecast_israel first."

    # סלקטור מדויק לשדה החיפוש באתר weather2day
    search_input_selector = "input#search_input, input[name='q'], input[type='search']"

    try:
        await _page.wait_for_selector(search_input_selector, timeout=5000)
        await _page.click(search_input_selector)
        await _page.fill(search_input_selector, city)
        await _page.wait_for_timeout(1000)
    except PlaywrightTimeout:
        return "Error: could not find the search field on the page."

    return f"Entered '{city}' into the search field."


@mcp.tool()
async def select_weather_forecast_city_israel() -> str:
    """Selects the first city suggestion from the dropdown list or submits search."""
    global _page

    if _page is None:
        return "Error: browser is not open. Call open_weather_forecast_israel first."

    suggestion_selector = ".easy-autocomplete-container li:first-child, .autocomplete-suggestions div:first-child"

    try:
        # ניסיון ללחוץ על ההצעה הראשונה במידה והיא מופיעה
        if await _page.is_visible(suggestion_selector):
            await _page.click(suggestion_selector)
        else:
            # במידה ואין רשימה נפתחת, לוחצים Enter בשדה החיפוש
            await _page.keyboard.press("Enter")
            
        await _page.wait_for_load_state("domcontentloaded")
        await _page.wait_for_timeout(2000)
    except PlaywrightTimeout:
        return "Error: could not select city suggestion."

    return "Selected the city. Forecast page should now be loaded."


@mcp.tool()
async def extract_weather_page_content_israel() -> str:
    """Extracts and cleans the weather forecast text from the currently loaded page after full rendering."""
    global _page

    if _page is None:
        return "Error: browser is not open."

    try:
        # המתנה לכך שהאלמנט המרכזי של התחזית או לפחות הגוף יכילו טקסט אמיתי ולא רק סקריפטים
        # מחכים שאלמנט הטקסט באתר יופיע (למשל טבלת התחזית או אזור התוצאות)
        await _page.wait_for_selector("div.forecast-box, div.main-content, body", timeout=10000)
        
מתן זמן קצר נוסף לוודא שסקריפטי ה-React סיימו לרנדר את הנתונים
        await _page.wait_for_timeout(2000)

        # שליפת הטקסט הנראה לעין בלבד מתוך אלמנט מרכזי או מהגוף
        text_content = await _page.inner_text("body")
        
        # סינון שורות ריקות וטקסט מיותר לחיסכון בטוקנים
        lines = [line.strip() for line in text_content.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines)
        
        # וידוא שקיבלנו תוכן אמיתי ולא רק קוד טכני
        if "_next/static" in cleaned_text or len(cleaned_text) < 50:
            return "Error: Page content is still loading or showing raw scripts. Please wait another moment."

        return cleaned_text[:3000]
    except Exception as e:
        return f"Error extracting page content: {str(e)}"

def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()