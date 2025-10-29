import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from pydantic import BaseModel
import datetime

from crewai_agent_examples.stock_agent_tools import store_stock_data, StockAnalysisData

# create agents
search_tool = SerperDevTool()

# agent for research task
stock_research_agent = Agent(
    role="Stock Research Agent",
    goal="Research financial markets and stock information for {market}",
    backstory=("You are a seasoned financial analyst with deep knowledge of stock markets and investment strategies."
              " Your expertise allows you to gather and analyze market data effectively."
              " Your goal is to gather comprehensive information on stocks and market trends."),
    tools=[search_tool],
    allow_delegation=True,
    verbose=True
)

# define tasks for stock research agent
research_task = Task(
    description="Conduct in-depth research on the current state of the {market} stock market.",
    expected_output="Comprehensive market data, stock performance metrics, and relevant news articles.",
    agent=stock_research_agent
)

# agent for stock analysis task
stock_analysis_agent = Agent(
    role = "Stock Analysis Agent",
    goal = "Analyze stock data and provide investment insights for {market}",
    backstory=(" With a strong background in financial analysis, you excel at interpreting stock data and market trends."
               " Your goal is to interpret the researched data and identify top {number} of stocks to buy."
               " Generate stock name, stock code, buy price, target price for both day and week trades, "
               " stop loss prices for each identified stock, current date and time as analysis date time,"
               " in the {market} and provide a brief rationale for each recommendation."),
    tools =[search_tool],
    allow_delegation=True,
    verbose=True,
    memory=True
)

# define tasks for stock analysis agent
analysis_task = Task(
    description="Analyze the researched stock data and identify top {number} stocks to buy with detailed recommendations.",
    expected_output=("A list of top {number} stocks along with stock code in the specified {market} to buy with buy price, "
                     "target price for day and weekly trades, stop loss prices,"
                     "analysis date time and rationale."),
    agent=stock_analysis_agent
)


# define agent to store indentified stock data
stock_data_storage_agent = Agent(
    role="Stock Data Storage Agent",
    goal="Store and manage researched stock data efficiently",
    backstory=("You are responsible for organizing and maintaining the integrity of stock data generated from stock_analysis_agent"
               " Your expertise ensures that all researched information is accurately stored and easily accessible."),
    tools=[store_stock_data]
)

# define pydantic model to map output of task


# define task for stock_data_storage_agent
storage_task = Task(
    description="Store the analyzed stock data into the database for future reference.",
    expected_output="Confirmation of successful data storage.",
    output_pydantic=StockAnalysisData,
    agent=stock_data_storage_agent
)


if __name__ == '__main__':
    # create crew to orchestrate the agents and tasks
    stock_crew = Crew(
        agents=[stock_research_agent, stock_analysis_agent, stock_data_storage_agent],
        tasks=[research_task, analysis_task, storage_task],
        verbose=True,
        memory=True
    )

    inputs = {
        "market": "Sweden",
        "number": 5
    }

    stock_crew.kickoff(inputs=inputs)


