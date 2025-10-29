from crewai import Agent, Task, Crew
from crewai_tools import YoutubeVideoSearchTool
from load_dotenv import load_dotenv

# Method # 1
# General search across Youtube content without specifying a video URL, so the agent can search within any Youtube video content it learns about irs url during its operation
#search_tool = YoutubeVideoSearchTool()
load_dotenv()
# Method # 2
# Targeted search within a specific Youtube video's content
search_tool = YoutubeVideoSearchTool(youtube_video_url='https://www.youtube.com/watch?v=R0ds4Mwhy-8')

# define agents
researcher = Agent(
    role="Video content researcher",
    goal="Extract key insights from youtube video",
    backstory=(
        "You are a skilled researcher who excels at extracting valuable insights from youtube video content"
        "You focus on gathering accurate and relevant information from Youtube to support your team"
    ),
    verbose=True,
    tools=[search_tool],
    memory =True
)

writer = Agent(
    role="Tech Article Writer",
    goal='Craft an article based on the research insights',
    backstory=(
        "You are an experienced writer known for turning complex information into engaging and accessible articles. "
        "Your work helps make advanced technology topics understandable to a broad audience."
    ),
    verbose=True,
    tools=[search_tool],
    memory=True
)

# define tasks
research_task = Task(
    description=(
        "Research and extract key insights from the given YouTube video regarding Educative. "
        "Compile your findings in a detailed summary."
    ),
    expected_output='A summary of the key insights from the YouTube video',
    agent=researcher
)

writer_task = Task(
description=(
        "Using the summary provided by the researcher, write a compelling article on what is Educative. "
        "Ensure the article is well-structured and engaging for a tech-savvy audience."
    ),
    expected_output='A well-written article on Educative based on the YouTube video research.',
    agent=writer,
    human_input=True # allow human input
)

if __name__ == '__main__':

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writer_task],
        verbose=True,
        memory=True
    )

    result = crew.kickoff()

    print("Execution completed")