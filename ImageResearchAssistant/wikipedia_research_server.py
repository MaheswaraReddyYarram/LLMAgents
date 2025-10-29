import wikipedia
from typing import List, Dict, Any
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context
load_dotenv()
import logging

# Initialize the FastMCP server
mcp = FastMCP("WikipediaResearchServer")

@mcp.tool()
async def fetch_wikipedia_info(query:str, ctx: Context, num_articles: int = 1 )-> List[Dict[str, Any]]:
    """"
    Fetches summaries and URLs of Wikipedia articles related to the given query.
    Args:
        query: The search term to look up on Wikipedia.
        num_articles: The number of top articles to retrieve.
    Returns:
        A list of dictionaries, each containing the title, summary, and URL of a Wikipedia article.
    """
    try:
        await ctx.info(f"Received request to fetch wikipedia info for query:{query}")
        print(f"Received request to fetch wikipedia info for query:{query}")
        # Get a list of potential page titles from the search query
        search_results = wikipedia.search(query, results=num_articles)
        if not search_results:
            await ctx.info(f"No articles found for query: {query}")
            return [{"error": f"No articles found for query: {query}"}]

        articles_info = []
        for title in search_results:
            try:
                # Retrieve page object for each title
                page = wikipedia.page(title, auto_suggest=False)
                articles_info.append({
                    "title": page.title,
                    "summary": page.summary.split('\n')[0],  # Get First paragraph as brief summary
                    "url": page.url
                })
            except wikipedia.DisambiguationError as e:
                # if title is ambigious, we will just skip it and try next one
                continue
            except wikipedia.PageError as e:
                # if page not found, skip to next
                continue

        if not articles_info:
            await ctx.info(f"No articles found for query: {query}")
            return [{"error": f"No valid articles found for query: {query}"}]

        return articles_info
    except Exception as e:
        await ctx.error(f"An unexpected error occurred:: {str(e)}")
        return [{"error": f"An unexpected error occurred: {str(e)}"}]

if __name__ == '__main__':
    logging.getLogger("mcp").setLevel(logging.DEBUG)
    mcp.run(transport="stdio")
