from crewai import Agent, Task, Crew

from crewai_agent_examples.research_hierarchy import data_analysis_task

coding_agent = Agent(
    role= "Python Data Analyst",
    goal="Write and execute python code to perform calculations",
    backstory=(
        "You are an experienced python developer, skilled at writing efficient code to solve problems"
    ),
    verbose=True,
    allow_code_execution=True
)

#define tasks
data_analysis_task = Task(
    description=(
        "Write Python code to calculate the average of the following list of ages: [23, 35, 31, 29, 40]. "
        "Output the result in the format: 'The average age of participants is: <calculated_average_age>'"
    ),
    expected_output="The generated code based on the requirements and the average age of participants is: <calculated_average_age>.",
    agent=coding_agent
)

# create debugging agent
debug_agent = Agent(
    role = "Python debugger",
    goal="identify and fix issues in python code",
    backstory=(
        "You are an experienced python developer with a knack for finding and fixing bugs"
    ),
    allow_code_execution=True
)

debug_task = Task(
    description=(
        "The following Python code is supposed to return the square of each number in the list, "
        "but it contains a bug. Please identify and fix the bug:\n"
        "```\n"
        "numbers = [2, 4, 6, 8]\n"
        "squared_numbers = [n*m for n in numbers]\n"
        "print(squared_numbers)\n"
        "```"
    ),
    agent=debug_agent,
    expected_output="The corrected code should output the squares of the numbers in the list. Provide the updated code and tell what was the bug and how you fixed it."

)

if __name__ == '__main__':
    analysis_crew = Crew(
        agents=[coding_agent],
        tasks=[data_analysis_task],
    )

    result = analysis_crew.kickoff()
    print(result)