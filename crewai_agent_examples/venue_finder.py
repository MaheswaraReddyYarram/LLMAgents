import os

from crewai import Crew, Agent, Task
from crewai_tools import SerperDevTool
from load_dotenv import load_dotenv
import agentops
agentops.init(api_key="YOUR_API_KEY")
load_dotenv()

agentops.init(api_key=os.getenv('AGENTOPS_API_KEY'))
# define tools
search_tool = SerperDevTool()
# crewai test --n_iterations 3
# define agents
venue_finder_agent = Agent(
    role = "Conference Venue Finder",
    goal = "Identify and recommend suitable venues for hosting conferences based on specified criteria",
    backstory=(
        "You are an experienced event planner with a knack for finding the perfect venues. "
        "Your expertise ensures that all conference requirements are met efficiently. "
        "Your goal is to provide the client with the best possible venue options."
    ),
    tools = [search_tool],
    verbose= True
)

venue_quality_assurance_agent = Agent(
    role = "Venue Quality Assurance Specialist",
    goal="Ensure the selected venues meet all quality standards and client requirements",
    backstory=(
        "You are meticulous and detail-oriented, ensuring that the venue options provided "
        "are not only suitable but also exceed the client's expectations. "
        "Your job is to review the venue options and provide detailed feedback."
    ),
    tools=[search_tool],
    verbose=True
)

#define tasks
find_venue_task = Task(
    description=("Conduct a thorough search to find the best venue for the upcoming "
        "conference. Consider factors such as capacity, location, amenities, "
        "and pricing. Use online resources and databases to gather comprehensive "
        "information."),
    expected_output="A detailed report listing 5 potential venues with detailed information on capacity, location, amenities, and pricing.",
    agent=venue_finder_agent,
    tools=[search_tool]
)

quality_review_task = Task(
    description=("Review the venue options provided by the Conference Venue Finder. "
        "Ensure that each venue meets all the specified requirements and standards. "
        "Provide a detailed report on the suitability of each venue."),
    expected_output=(
        "A detailed review of the 5 potential venues, highlighting any issues, strengths, and overall suitability."
    ),
    tools=[search_tool],
    agent=venue_quality_assurance_agent
)

#define crew
event_planning_crew = Crew(
    agents=[venue_finder_agent, venue_quality_assurance_agent],
    tasks=[find_venue_task, quality_review_task],
    verbose=True,
    memory=True
)

if __name__ == '__main__':
    inputs = {
        "conference_name": "Tech Innovators Summit 2024",
        "requirements": "Capacity for 5000, central location, modern amenities, budget up to $50,000"
    }

    result = event_planning_crew.kickoff(inputs=inputs)
    print("Final Output:\n", result.output if hasattr(result, 'output') else str(result))