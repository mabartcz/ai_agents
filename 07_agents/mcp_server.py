from mcp.server.fastmcp import FastMCP
from langchain_community.tools import BraveSearch

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def search_web(input: str) -> int:
    """Search the web for result"""
    tool = BraveSearch.from_search_kwargs(search_kwargs={"count": 1})
    search_data = tool.run(input)
    return search_data

if __name__ == "__main__":
    mcp.run(transport="stdio")