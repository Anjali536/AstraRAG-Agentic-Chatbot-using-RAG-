from crewai import Agent

researcher = Agent(
    role="Researcher",
    goal="Find relevant information from documents",
    backstory="Expert in retrieving information from knowledge bases.",
    verbose=True
)