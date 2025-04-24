#!/usr/bin/env python
from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from test_flow.crews.poem_crew.pr_crew import PRCrew


# State class to track the PR analysis flow
class PRState(BaseModel):
    company: str = "groq"  # Default company to analyze
    articles: str = ""          # Will store collected articles
    analysis: str = ""          # Will store the analysis
    report: str = ""            # Will store the final report
    company_goals: str = ""     # Will store company PR goals


# Dictionary of company PR goals
COMPANY_GOALS = {
    "groq": """
    1. Position Groq as the fastest and most efficient LLM inference platform
    2. Highlight the technical superiority of GroqChip and LPU architecture
    3. Establish Groq as a leader in AI infrastructure for enterprise applications
    4. Demonstrate real-world performance advantages over GPU-based solutions
    5. Build developer community and ecosystem around Groq technology
    """,
    
    "microsoft": """
    1. Maintain leadership position in enterprise cloud and AI solutions
    2. Promote integration of AI across Microsoft's product portfolio
    3. Emphasize commitment to responsible AI development and deployment
    4. Highlight strategic partnerships with OpenAI and other AI leaders
    5. Position Microsoft as an innovator in the future of work and productivity
    """,
    
    "openai": """
    1. Lead the global narrative on AI alignment and safety
    2. Promote transparency and responsible LLM usage
    3. Reassure public about AI control and ethical use
    4. Highlight partnerships and use cases
    5. Mitigate concerns around hallucinations and misuse
    """,
    
    "anthropic": """
    1. Differentiate Claude as the most helpful, harmless, and honest AI assistant
    2. Emphasize Anthropic's commitment to AI safety and Constitutional AI
    3. Position Anthropic as a thoughtful, research-driven organization
    4. Highlight enterprise partnerships and Claude's business applications
    5. Build trust through transparency in AI development practices
    """
}


class PRFlow(Flow[PRState]):

    @start()
    def collect_company_articles(self):
        print(f"Collecting articles about {self.state.company}")
        
        # Set company goals based on the company name
        company_key = self.state.company.lower()
        self.state.company_goals = COMPANY_GOALS.get(
            company_key, 
            f"No specific PR goals defined for {self.state.company}. Analyze general PR performance."
        )
        
        # Run the PR crew exactly like in the poem example
        print("Starting PR analysis...")
        
        # This will throw an error if it fails - no fallback
        result = (
            PRCrew()
            .crew()
            .kickoff(inputs={
                "company": self.state.company,
                "company_goals": self.state.company_goals,
                "current_year": "2025"
            })
        )
        
        print("PR analysis completed")
        # Store the result for the next steps
        self.state.report = result.raw
        
        # Extract articles and analysis from the result if possible
        if "Articles collected" in result.raw:
            self.state.articles = result.raw.split("Articles collected:")[1].split("Analysis:")[0].strip()
        else:
            self.state.articles = f"[Collected articles about {self.state.company}]"
            
        if "Analysis:" in result.raw:
            self.state.analysis = result.raw.split("Analysis:")[1].split("Recommendations:")[0].strip()
        else:
            self.state.analysis = f"Analysis of {self.state.company} PR performance"

    @listen(collect_company_articles)
    def save_report(self):
        print("Saving PR report")
        report_filename = f"{self.state.company.lower()}_pr_report.md"
        with open(report_filename, "w") as f:
            f.write(self.state.report)
        print(f"Report saved to {report_filename}")


def kickoff(company="groq"):
    pr_flow = PRFlow(state=PRState(company=company))
    pr_flow.kickoff()


def plot():
    pr_flow = PRFlow()
    pr_flow.plot()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If a company name is provided as a command-line argument, use it
        company_name = sys.argv[1]
        kickoff(company_name)
    else:
        # Otherwise use the default (groq)
        kickoff()
