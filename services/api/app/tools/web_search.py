"""
Web search tool: queries the Tavily search API for current information not covered by the document corpus.
"""
import httpx
import os


async def web_search_tool(query: str) -> str:
    """Searches the web via Tavily.

    Args:
        query: The search query.

    Returns:
        Formatted search results, a message if none were found, or an error/disabled message.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search is disabled (API Key missing)."

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 3
                },
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            formatted = "\n".join([f"- {r['title']}: {r['content']} ({r['url']})" for r in results])

            return formatted if formatted else "No results found on the web."

    except Exception as e:
        return f"Web Search Error: {str(e)}"