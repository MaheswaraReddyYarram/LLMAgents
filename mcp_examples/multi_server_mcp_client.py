import asyncio
import shlex

from mcp import StdioServerParameters
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
# Import the MultiServerMCPClient
from langchain_mcp_adapters.client import MultiServerMCPClient

# --- Multi-server configuration dictionary ---
# This dictionary defines all the servers the client will connect to
server_configs = {
    "weather": {
        "command": "python",
        "args": ["weather_server.py"],  # Your original weather server
        "transport": "stdio",
    },
    "tasks": {
        "command": "python",
        "args": ["task_server.py"],  # The new task management server
        "transport": "stdio",
    }
}
load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

# LangGraph state definition (remains the same)
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


# --- 'create_graph' now accepts the list of tools directly ---
def create_graph(tools: list):
    # LLM configuration (remains the same)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=google_api_key)
    llm_with_tools = llm.bind_tools(tools)

    # --- Updated system prompt to reflect new capabilities ---
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. You have access to tools for checking the weather and managing a to-do list. Use the tools when necessary based on the user's request."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    # Define chat node (remains the same)
    def chat_node(state: State) -> State:
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    # Build LangGraph with tool routing (remains the same)
    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())


# --- Main function now uses MultiServerMCPClient ---
async def tool_main():
    # As per the error message, instantiate the client directly
    # The client will manage the server subprocesses internally
    client = MultiServerMCPClient(server_configs)

    # Get a single, unified list of tools from all connected servers
    all_tools = await client.get_tools()

    # Create the LangGraph agent with the aggregated list of tools
    agent = create_graph(all_tools)

    print("MCP Agent is ready (connected to Weather and Task servers).")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        try:
            response = await agent.ainvoke(
                {"messages": [("user", user_input)]},
                config={"configurable": {"thread_id": "multi-server-session"}}
            )
            print("AI:", response["messages"][-1].content)
        except Exception as e:
            print("Error:", e)

async def list_all_prompts(client: MultiServerMCPClient, server_configs: dict):
    print("Available prompts from all servers:")
    print("-----------------------------------")
    any_prompts_found = False

    # Iterate through the names of servers
    for server_name in server_configs.keys():
        try:
            # Opening session for a specific server
            async with client.session(server_name) as session:
                prompt_response = await session.list_prompts()
                if prompt_response and prompt_response.prompts:
                    any_prompts_found = True
                    # Print a header for the server to group the prompts
                    print(f"\n--- Server '{server_name}'---")
                    for p in prompt_response.prompts:
                        print(f"- Prompt ---{p.name}")
                        if p.arguments:
                            arg_list =  [f"{arg.name} " for arg in p.arguments]
                            print(f"    Arguments: {', '.join(arg_list)}")
                        else:
                            print("     Arguments: None")
        except Exception as e:
            print(f"\nCould not fetch prompts from server '{server_name}': {e}")

    print(f"\nUse: /prompt <server_name> <prompt_name> \"arg1\" \"arg2\"...")
    print("-----------------------------------")
    if not any_prompts_found:
        print("\n No prompts were found on any connected server")

async def handle_prompt_invocation(client: MultiServerMCPClient, command: str) -> str| None:
    try:
        #use shlex to parse the command string
        parts = shlex.split(command.strip())

        # command should be :/prompt <server_name> <prompt_name> [args...]
        if(len(parts) < 3):
            print("Usage: /prompt <server_name> <prompt_name> [args...]")
            return None

        server_name = parts[1]
        prompt_name = parts[2]
        user_args = parts[3:]

        # validate the prompt and arguments against the specific server
        prompt_def = None
        async with client.session(server_name) as session:
            all_prompt_response = await session.list_prompts()
            if all_prompt_response and all_prompt_response.prompts:
                # find matching prompt name
                prompt_def = next((p for p in all_prompt_response.prompts if p.name == prompt_name), None)

        if not prompt_def:
            print(f"Prompt '{prompt_name}' not found on server '{server_name}'.")
            return None

        # check if number of user provided arguments matches the prompt definition
        if len(user_args) != len(prompt_def.arguments):
            expected_args = [arg.name for arg in prompt_def.arguments]
            print(f"\nError: Invalid number of arguments for prompt '{prompt_name}'.")
            print(f"Expected {len(expected_args)} arguments: {', '.join(expected_args)}")
            return None

        #fetch and execute the prompt
        arg_dict = {arg.name: val for arg,val in zip(prompt_def.arguments, user_args)}

        # use client's get_prompt method
        prompt_messages = await client.get_prompt(server_name, prompt_name, arg_dict)

        #get prompt text
        prompt_text = prompt_messages[0].content

        print(f"Prompt loaded successfully, Invoking the prompt")

        return prompt_text

    except Exception as e:
        print(f"Error handling prompt invocation: {e}")
        return None


async def promtpt_main():
    # Instantiate the client which will manage the server subprocesses internally
    client = MultiServerMCPClient(server_configs)

    # Get a single, unified list of tools from all connected servers
    all_tools = await client.get_tools()

    # Create the LangGraph agent with the aggregated list of tools
    agent = create_graph(all_tools)

    print("MCP Agent is ready (connected to Weather and Task servers).")
    # Update the instructions for the user
    print("Type a question, or use one of the following commands:")
    print("  /prompts                                       - to list available prompts")
    print("  /prompt <server_name> <prompt_name> \"args\"   - to run a specific prompt")

    while True:
        # This variable will hold the final message to be sent to the agent
        message_to_agent = ""

        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break

        # --- Command Handling Logic ---
        if user_input.lower() == "/prompts":
            await list_all_prompts(client, server_configs)
            continue  # Command is done, loop back for next input

        elif user_input.startswith("/prompt"):

            # The function returns the prompt text or None.
            prompt_text = await handle_prompt_invocation(client, user_input)
            if prompt_text:
                message_to_agent = prompt_text
            else:
                # If prompt fetching failed, loop back for next input
                continue

        else:
            # For a normal chat message, the message is just the user's input
            message_to_agent = user_input

        # --- Final agent invocation ---

        if message_to_agent:
            try:
                response = await agent.ainvoke(
                    {"messages": [("user", message_to_agent)]},
                    config={"configurable": {"thread_id": "multi-server-session"}}
                )
                print("AI:", response["messages"][-1].content)
            except Exception as e:
                print("Error:", e)

if __name__ == "__main__":
    #asyncio.run(tool_main())
    asyncio.run(promtpt_main())