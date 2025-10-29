from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from load_dotenv import load_dotenv
import os

load_dotenv()

search_tool = SerperDevTool()
scrape_tool= ScrapeWebsiteTool()

research_analyst_agent = Agent(
    role = "Research Analyst",
    goal = "Create and analyze research points to provide comprehensive insights on various topics",
    backstory=(
        "Specializing in research analysis, this agent employs advanced methodologies to generate detailed research points "
        "and insights. With a deep understanding of research frameworks and a talent for synthesizing information, "
        "the Research Analyst Agent is instrumental in delivering thorough and actionable research outcomes."
    ),
    verbose=True,
    memory=True,
    allow_delegation=True,
    tools= [scrape_tool, search_tool]
)

report_writer_agent = Agent(
    role="Report Writer",
    goal="Compile the analyzed data into a comprehensive and well-structured research report.",
    backstory="You are skilled at transforming complex information into clear, concise, and informative reports.",
    verbose=True,
    allow_delegation=True
)

report_editor_agent = Agent(
    role="Report Editor",
    goal="Review and refine research reports to ensure clarity, accuracy, and adherence to standards.",
    backstory=("With a keen eye for detail and a strong background in report editing, this agent ensures that research "
              "reports are polished, coherent, and meet high-quality standards. Skilled in revising content for clarity "
              "and consistency, the Report Editor Agent plays a critical role in finalizing research outputs."),
    verbose=True
)

# define tasks
data_collection_task = Task(
    description=( "Collect data from relevant sources about the given {topic}."
        "Focus on identifying key trends, benefits, and challenges."),
    expected_output=(
        "A comprehensive dataset that includes recent studies, statistics and expert opinions"
    ),
    agent=research_analyst_agent
)

data_analysis_task = Task(
    description=(
        "Analyze the collected data to identify key trends, benefits, and challenges for the {topic}."
    ),
    expected_output=(
        "A detailed analysis report highlighting the most significant findings."
    ),
    agent=research_analyst_agent,
)

report_writing_task = Task(
    description=(
        "Write a comprehensive research report that clearly presents the findings from the data analysis report"
    ),
    expected_output=(
        "A well-structured research report that provides insights about the topic."
    ),
    agent=report_writer_agent,
)

report_assessment_task = Task(
    description=(
        "Review and rewrite the research report to ensure clarity, accuracy, and adherence to standards."
    ),
    expected_output=(
        "A polished, coherent research report that meets high-quality standards and effectively communicates the findings."
    ),
    agent=report_editor_agent,
)

from langchain_groq import ChatGroq

model = 'llama3-groq-70b-8192-tool-use-preview'
llm = ChatGroq(
            temperature=0,
            groq_api_key = os.getenv('GROQ_API_KEY'),
            model_name=model
        )

# define hierarchial crew with management llm
research_crew = Crew(
    agents = [research_analyst_agent, report_writer_agent, report_editor_agent],
    tasks= [data_collection_task, data_analysis_task, report_writing_task, report_assessment_task],
    manager_llm=llm,
    process=Process.hierarchical,
    verbose=True
)
