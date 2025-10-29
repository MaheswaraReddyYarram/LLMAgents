import asyncio
from typing import List
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

async def main():
    """
    Main function to set up llama index agent
    :return:
    """
    print("Intializing LlamaIndex Agent...")

    #setup llm
    llm = GoogleGenAI(model_name = 'gemini-1.5-flash', google_api_key=google_api_key)

    #set up client
    mcp_client = BasicMCPClient("python", args=["weather_server.py"])

    #set up tool spec
    tool_spec = McpToolSpec(client=mcp_client)

    # The agent will use the tools loaded from the MCP server
    # We use the async method to fetch the tool definitions
    mcp_tools:List = await tool_spec.to_tool_list_async()
    print(f"Successfully loaded {len(mcp_tools)} tools from MCP server.")

    # create llama index agent
    agent = ReActAgent(tools=mcp_tools, llm=llm, verbose=True)

    print("\n Weather MCP Agent is ready.Ask for weather (e,g., 'What's the weather in New York?')\n")

    #start conversation loop
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Exiting...")
            break

        if not user_input:
            print("Please enter a valid input.")
            continue
        try:
            # agent's chat method handles full reasoning and tool-calling loop
            response = await agent.run(user_input)
            print("\nAgent:", str(response))
        except Exception as e:
            print(f"Error during agent execution: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main())