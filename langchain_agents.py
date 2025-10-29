from load_dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import OpenAI
import os

load_dotenv()

if __name__ == '__main__':
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = 'Project Name'

    # get prompt to use
    prompt = hub.pull('hwchase17/react')
    prompt = """
    Answer the following questions as best you can. You have access to the following tools:,

{tools}

Use the following format:

Question: the input question you must answer

Thought: you should always think about what to do

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question

Begin!

Question: {input}

Thought:{agent_scratchpad}"""

    # define tools to use
    tools  = [TavilySearchResults(max_results = 1)]

    # define llm to use
    llm = OpenAI()

    # create agent
    agent = create_react_agent(llm, tools, prompt)

    # create agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": "What is Educative?"})

    print(response)