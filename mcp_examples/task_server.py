import os

from anyio import TASK_STATUS_IGNORED
from mcp.server.fastmcp import FastMCP
from typing import List

# Define the file where tasks will be stored
TASKS_FILE = 'tasks.txt'

# Initialize server
mcp = FastMCP("TaskManagerAssistant")

@mcp.tool()
def add_task(task_description: str) -> str:
    """
    Adds a new task to the persistent task list file.

    This tool will create the task file if it doesn't exist. It appends
    the new task to the end of the file, ensuring each task is on a new line.

    Args:
        task_description: A string describing the task to be added.
                          For example: "Buy groceries" or "Finish the report".

    Returns:
        A string confirming that the task was successfully added.
    """
    try:
        # 'a' mode will append to the file, and create it if it doesn't exist
        with open(TASKS_FILE, 'a') as f:
            f.write(task_description + '\n')
        return f"Task added: {task_description}"
    except Exception as e:
        return f"Error while adding task: {str(e)}"

@mcp.tool()
def list_tasks() -> List[str]:
    """
    Lists all tasks from the persistent task list file.

    This tool reads the tasks from the file and returns them as a list of strings.
    If the file does not exist, it returns an empty list.

    Returns:
        A list of strings, each representing a task. If no tasks are found,
        returns an empty list.
    """
    if not os.path.exists(TASKS_FILE):
        # Return empty list if the file does not exist
        return []

    try:
        with open(TASKS_FILE, 'r') as f:
            # Read all lines, strip leading/trailing sequences, and filter out empty lines
            tasks = [line.strip() for line in f.readlines()]
            # filter out empty lines
            return [task for task in tasks if task]
        return tasks
    except Exception as e:
        # In case of an error, we can return a list with an error message,
        # but for simplicity and better type consistency, we'll return an empty list.
        # The LLM can be prompted to handle this gracefully
        print(f"Error while listing tasks: {str(e)}")
        return []

@mcp.prompt()
def plan_trip_prompt(destination: str, duration_in_days: int) -> str:
    """
        Creates a sample travel itinerary for a given destination and duration, then saves it as a series of tasks in the user's to-do list.
        This is the best prompt to use when a user asks to plan a trip, create an itinerary, or asks for travel suggestions.

        Args:
            destination: The city or country for the trip (e.g., "Paris", "Japan").
            duration_in_days: The number of days for the trip (e.g., 3).
        """
    return f"""
        You are an expert travel consultant. Your goal is to help the user by generating a sample travel itinerary and saving it to their task list for later reference.

        The user wants a plan for a {duration_in_days}-day trip to {destination}.

        Follow these steps carefully:
        1.  First, use your general knowledge to brainstorm a simple, day-by-day itinerary. Suggest one or two key attractions or activities for each day of the trip.
        2.  After you have formulated the plan, you MUST perform a critical action: for each individual activity or attraction in your suggested itinerary, save it to the user's task list. For example, if you suggest visiting the Louvre, you must call the tool for that specific item.
        3.  Once all the itinerary items have been added as tasks, present a friendly confirmation message to the user. Inform them that you have created a sample plan and saved it to their to-do list.
        """


if __name__ == '__main__':
    print(f'Starting MCP Task server ')
    mcp.run(transport="stdio")