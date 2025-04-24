import os
from dotenv import load_dotenv
import openai
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool

# === Load keys ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# === Hardcoded values ===
company_name = "OpenAI"  # Change this value to analyze a different company
effort_level = 5  # Number of articles to analyze

# === Hardcoded PR Goals ===
company_goals_dict = {
    "openai": """
    1. Lead the global narrative on AI alignment and safety.
    2. Promote transparency and responsible LLM usage.
    3. Reassure public about AI control and ethical use.
    4. Highlight partnerships and use cases.
    5. Mitigate concerns around hallucinations and misuse.
    """
}
company_goals = company_goals_dict.get(company_name.lower(), "No PR goals available.")

# === Define Serper Tool ===
search_tool = SerperDevTool()

# === Define Agents ===
researcher = Agent(
    role="Web Researcher",
    goal="Find recent articles and online discussions about the company",
    backstory="You are an elite web investigator skilled at using search tools to uncover public discourse and recent news.",
    tools=[search_tool],
    verbose=True
)

analyst = Agent(
    role="Sentiment Analyst",
    goal="Analyze the tone and sentiment of news articles",
    backstory="You are a sentiment specialist who can detect positive and negative mood, sarcasm, and emotional tone in articles.",
    verbose=True
)

evaluator = Agent(
    role="PR Goals Evaluator",
    goal="Evaluate how well each article aligns with the company's PR goals",
    backstory="You are a strategic PR consultant who assesses media coverage against organizational objectives.",
    verbose=True
)

# === Define Tasks ===
research_task = Task(
    description=f"""
    Use your search tools to find up to {effort_level} recent news articles about "{company_name}".
    Return a list of dictionaries like: 
    [{{"title": "...", "link": "...", "snippet": "..."}}]
    """,
    expected_output="List of articles with title, snippet, and link.",
    agent=researcher
)

sentiment_task = Task(
    description="""
    For each article, perform sentiment analysis.
    Return a list of dicts with:
    - title
    - mood (e.g., upbeat, critical, neutral)
    - positivity_score (1 to 10)
    """,
    expected_output="List of dicts like [{'title': ..., 'mood': ..., 'score': ...}]",
    agent=analyst
)

goal_evaluation_task = Task(
    description=f"""
    For each article, evaluate how well it aligns with the company's PR goals:
    {company_goals}
    
    For each article, provide:
    1. Title of the article
    2. A score (1-10) for how well the article aligns with EACH of the company's goals
    3. Brief explanation of why you gave each score
    4. Overall alignment score (1-10)
    
    Return your analysis in a clear, structured format.
    """,
    expected_output="Detailed evaluation of each article against company PR goals",
    agent=evaluator
)

# === Run the Crew ===
print(f"\n🎯 Company: {company_name}")
print(f"📌 Goals: {company_goals}\n")

crew = Crew(
    agents=[researcher, analyst, evaluator],
    tasks=[research_task, sentiment_task, goal_evaluation_task],
    verbose=True
)

results = crew.kickoff()
print(f"\nResults: {results}")
