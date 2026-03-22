from crewai import Agent

analyst = Agent(
    role="Analyst",
    goal="Analyze the retrieved information carefully",
    backstory="Expert at breaking down complex information.",
    verbose=True
)