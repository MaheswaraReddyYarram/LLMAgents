import asyncio
import os
import shlex
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.tools import load_mcp_tools

# MCP server configuration
# server_params = StdioServerParameters(
#     command = "/Users/mahyaa/PycharmProjects/NeuralNetworks/mcp_examples/mcp_venv/bin/python",
#     args =["/Users/mahyaa/PycharmProjects/NeuralNetworks/mcp_examples/single_server_mcp/weather_server.py"]
# )

server_params = StdioServerParameters(
    command = "python",
    args =["weather_server.py"]
)

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')
#Lang graph state definition
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

async def create_graph(session):
    # Load tools from the MCP server
    tools = await load_mcp_tools(session)
    print(f"Loaded {len(tools)} tools from MCP server.")
    print(f'tools: {tools}')

    # LLM configuration
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0,
                                 google_api_key = google_api_key)
    llm_with_tools = llm.bind_tools(tools)

    # prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that uses tools to get the current weather information for a location."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    #define chat node
    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        print(f'response: {state["messages"]}')
        return state

    # build lang graph with tool routing
    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition,{"tools": "tool_node", "__end__": END})
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())

async def list_prompts(session):
    """
    Fetches list of available prompts from the MCP server.
    and prints them in user-friendly manner
    :param session:
    :return:
    """
    try:
        prompt_response = await session.list_prompts()
        if not prompt_response or not prompt_response.prompts:
            print("\nNo prompts found or invalid response format.")
            return

        print("\nAvailable Prompts and their arguments:")
        print("----------------------------------------")
        for p in prompt_response.prompts:
            print(f"\nPrompt Name: {p.name}")
            if p.arguments:
                arg_list =[f"<{arg.name}>" for arg in p.arguments]
                print(f"Arguments: {', '.join(arg_list)}")
            else:
                print("Arguments: None")

        print("\nUsage: /prompt <prompt_name> \"arg1\" \"arg2\" ...")
        print("---------------------------------------")
    except Exception as e:
        print(f"Error fetching prompts: {e}")


async def handle_prompt(session, command:str) -> str | None:
    """
    Handles the /prompt command to invoke a specific prompt on the MCP server.
    :param session:
    :param user_input:
    :return:
    """
    try:
        parts = shlex.split(command.strip())
        if(len(parts)<2):
            print("\nUsage: /prompt <prompt_name> \"arg1\" \"arg2\" ...")
            return None
        prompt_name = parts[1]
        user_args = parts[2:]

        # get available prompts from the server to validate against
        prompt_def_response = await session.list_prompts()
        if not prompt_def_response or not prompt_def_response.prompts:
            print("\nError: Could not retrieve any prompts from the server.")
            return None

        # find specific format definition the user is asking for
        prompt_def = next((p for p in prompt_def_response.prompts if p.name == prompt_name), None)

        if not prompt_def:
            print(f"\nError: Prompt '{prompt_name}' not found on the server.")
            return None

        #check if number of user provided arguments match with the prompt expected arguments
        if len(user_args) != len(prompt_def.arguments):
            expected_arguments = [arg.name for arg in prompt_def.arguments]
            print(f"\nError: Invalid number of arguments for prompt '{prompt_name}'.")
            print(f"Expected arguments:{len(expected_arguments)}  arguments: {','.join(expected_arguments)}")
            return None

        #build argument dictionary
        arg_dict = {arg.name: val for arg,val  in zip(prompt_def.arguments, user_args)}

        # fetch the prompt from the server
        prompt_response = await session.get_prompt(prompt_name, arg_dict)

        # extract text content from the response
        prompt_text = prompt_response.messages[0].content.text

        print("\n--- Prompt loaded successfully. Preparing to execute... ---")
        # Return the fetched text to be used by the agent
        return prompt_text
    except Exception as e:
        print(f"Error handling prompt command: {e}")
        return None


#entry point for prompt
async def prompt_main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            agent = await create_graph(session)
            print(f'agent :{agent.__repr__()}')
            print("Weather MCP Agent is ready...")
            # list available prompts on startup
            print("\n Type a question or use any of the following commands:")
            print("\n     /prompts                               - to list available prompts from the server")
            print("\n     /prompt <prompt-name> \"args\"...      - to list available prompts from the server")


            while True:
                # This variable will hold the final message to be sent to the agent
                message_to_agent = ""

                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("Exiting...")
                    break

                # --- Command handling logic---
                if user_input.lower() == "/prompts":
                    await list_prompts(session)
                    continue  # command is done loop back for next input

                elif user_input.lower().startswith("/prompt"):
                    # The handle_prompt function now returns the prompt text or None
                    prompt_text = await handle_prompt(session, user_input)
                    if prompt_text:
                        message_to_agent = prompt_text
                    else:
                        # if prompt fetching failed , loop back for next input
                        continue
                else:
                    # for a normal chat message , use it as-is
                    message_to_agent = user_input

                # Final agent invocation
                # All paths (regular chat or successful prompt) now lead to this single block
                if message_to_agent:
                    try:
                        print(f'user_input: {message_to_agent}')
                        response = await agent.ainvoke(
                            {"messages": [("user", message_to_agent)]},
                                  config = {"configurable": {"thread_id": "weather-session"}},
                                  print_mode="debug"
                        )
                        print("Assistant:", response["messages"][-1].content)
                    except Exception as e:
                        print("Error:", e)


#entry point for tool
async def tool_main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            agent = await create_graph(session)
            print(f'agent :{agent.__repr__()}')
            print("Weather MCP Agent is ready...")

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("Exiting...")
                    break
                try:
                    print(f'user_input: {user_input}')
                    response = await agent.ainvoke({"messages": [("user", user_input)]},
                                                  config = {"configurable": {"thread_id": "weather-session"}},
                                                   print_mode="debug"
                                                  )
                    print("Assistant:", response["messages"][-1].content)
                except Exception as e:
                    print("Error:", e)


#resources code
async def list_resources(session):
    """
    Fetches list of available resources from the MCP server.
    and prints them in user-friendly manner
    :param session:
    :return:
    """
    try:
        resource_response = await session.list_resources()
        if not resource_response or not resource_response.resources:
            print("\nNo resources found on the server.")
            return

        print("\nAvailable Resources and their arguments:")
        print("----------------------------------------")
        for r in resource_response.resources:
            print(f"\nResource URI: {r.uri}")
            if r.description:
                print(f"Description: {r.description.strip()}")
            else:
                print("Arguments: None")

        print("\nUsage: /resource <resource_uri> ...")
        print("---------------------------------------")
    except Exception as e:
        print(f"Error fetching resources: {e}")

async def handle_resource(session, command:str) -> str | None:
    """
    Parses a user command to fetch a specific resource from the server
    and returns its content as a single string.
    :param session:
    :param user_input:
    :return:
    """
    try:
        parts = shlex.split(command.strip())
        if(len(parts) != 2):
            print("\nUsage: /resource <resource_uri> ...")
            return None
        resource_uri = parts[1]

        print(f'---Fetching resource: {resource_uri}.... ----')

        # Use the session's `read_resource` method with the provided URI
        response = await session.read_resource(resource_uri)

        if not response or not response.contents:
            print("\nError: Resource not found or content is empty.")
            return None

        # Extract text from all TextContent objects and join them
        # This handles cases where a resource might be split into multiple parts
        text_parts = [
            content.text for content in response.contents if hasattr(content, 'text')
        ]

        if not text_parts:
            print(f"Error: Resource content is not in readable format")
            return None

        resource_content = "\n".join(text_parts)
        print("\n--- Resource loaded successfully. Preparing to execute... ---")
        return resource_content
    except Exception as e:
        print(f":An error occurred while fetching the resource {e}")
        return None


async def resources_main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            agent = await create_graph(session)
            print("Weather MCP Agent is ready...")
            # Add instructions for resource commands
            print("Type a question, or use one of the following commands:")
            print("  /resources                       - to list available resources")
            print("  /resource <resource_uri>         - to load a resource for the agent")

            while True:
                # This variable will hold the final message to be sent to the agent
                message_to_agent = ""

                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    print("Exiting...")
                    break

                # --- Command handling logic---
                if user_input.lower() == "/resources":
                    await list_resources(session)
                    continue # Command is done, loop back for next input

                elif user_input.lower().startswith("/resource"):
                    # Fetch the resource from the server
                    resource_content = await handle_resource(session, user_input)

                    if resource_content:
                        # Ask the user what action to take on the loaded content
                        action_prompt = input("\nResource loaded. What would you like to do with it? (press Enter to just save the context): ").strip()

                        #If user provides an action prompt, prepend it to the resource content
                        if action_prompt:
                            message_to_agent = f"""
                            CONTEXT from a loaded resource
                            -----------------------------
                            {resource_content}
                            --
                            TASK: {action_prompt}
                            """
                        # If user provides no action, create a default message to save the context
                        else:
                            print("No action specified. Adding resource content to conversation memory...")
                            message_to_agent = f"""
                            Please remember the following context for our conversation. Just acknowledge that you have received it.
                            -----------------------------
                            CONTEXT:{
                            resource_content}
                            --
                            
                            """
                    else:
                        # If resource fetching failed, loop back for next input
                        continue
                else:
                    # For a normal chat message, use it as-is
                    message_to_agent = user_input

                # Final agent invocation
                # All paths (regular chat or successful resource load) now lead to this single block
                if message_to_agent:
                    try:
                        print(f'user_input: {message_to_agent}')
                        response = await agent.ainvoke(
                            {"messages": [("user", message_to_agent)]},
                                  config = {"configurable": {"thread_id": "weather-session"}},
                                  print_mode="debug"
                        )
                        print("Assistant:", response["messages"][-1].content)
                    except Exception as e:
                        print("Error:", e)



if __name__ == '__main__':
    load_dotenv()
    #asyncio.run(tool_main())
    #asyncio.run(prompt_main())
    asyncio.run(resources_main())

