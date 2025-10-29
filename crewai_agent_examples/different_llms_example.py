import os
from crewai_tools import SerperDevTool
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool, tool


search_tool = SerperDevTool()

# Initialize gemini model
gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    verbose = True,
    google_api_key="YOUR_GOOGLE_API_KEY"
)

# Initialize gpt-4 model
gpt = ChatOpenAI(
    model = "gpt-4o-2024-08-06",
    temperature=0.5,
    verbose=True,
    openai_api_key = os.getenv("OPENAI_API_KEY")
)


#create research agent using gemini
article_searcher = Agent(
    role = "Senior Researcher",
    goal = "Uncover ground breaking technologies in {topic}",
    verbose=True,
    memory=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of"
        "innovation, eager to explore and share knowledge that could change"
        "the world."
    ),
    tools = [search_tool],
    llm=gemini,
    allow_delegation=True
)

# create writer agent using gpt
article_writer = Agent(
    role="Writer",
    goal="Narate compelling tech stories about {topic}",
    verbose=True,
    memory=True,
    backstory= ("With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."),
    tools=[search_tool],
    llm=gpt,
    allow_delegation=False
)

# define custom tool
class MyCustomTool(BaseTool):
    name: str = 'MyCustomTool'
    description: str = "Clear description for what this tool is useful for, your agent will need this information to use it"

    def _run(self, argument:str) -> str:
        return "result"

@tool("Name of my tool")
def my_tool(question:str) -> str:
    """
    clear description of tool
    :param question:
    :return:
    """
    return "result"

# caching in tool
@tool("multiplication_tool")
def multiplication_tool(first: int, second:int)->int:
    """
    Useful when you need to multiple two numbers
    :param first:
    :param second:
    :return:
    """
    return first * second

def  cache_function(args, result):
    cache = result%2 ==0
    return cache

multiplication_tool.cache_function = cache_function()