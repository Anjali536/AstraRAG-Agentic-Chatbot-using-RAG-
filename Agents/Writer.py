from crewai import Agent

writer = Agent(
    role="Writer",
    goal="Generate a clear and helpful answer for the user",
    backstory="Expert technical writer.",
    verbose=True
)