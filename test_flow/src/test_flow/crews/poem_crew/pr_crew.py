from crewai import Agent, Crew, Process, Task

class PRCrew:
    """PR Crew for analyzing company PR performance"""

    def crew(self):
        """Create a crew for PR analysis"""
        
        # Create the agents
        company_pr_analyst = Agent(
            role="PR Analyst",
            goal="Find and collect recent articles about the company",
            backstory="You are an experienced PR analyst with expertise in finding and collecting relevant media coverage.",
            verbose=True,
            allow_delegation=False
        )
        
        pr_professional = Agent(
            role="PR Professional",
            goal="Analyze articles against company PR goals",
            backstory="You are a seasoned PR professional who can analyze media coverage and determine how well it aligns with company goals.",
            verbose=True,
            allow_delegation=False
        )
        
        report_writer = Agent(
            role="Report Writer",
            goal="Write comprehensive PR analysis reports",
            backstory="You are a skilled writer who specializes in creating clear, actionable PR analysis reports.",
            verbose=True,
            allow_delegation=False
        )
        
        # Create the tasks
        collect_articles_task = Task(
            description="Collect recent articles about {company}. Focus on news, press releases, and media coverage from the past month. Search for at least 5 articles.",
            expected_output="A list of at least 5 articles about the company with title, source, date, and a brief summary of each.",
            agent=company_pr_analyst
        )
        
        analyze_articles_task = Task(
            description="Analyze the collected articles about {company} against their PR goals: {company_goals}. Determine how well the company's media coverage aligns with these goals.",
            expected_output="A detailed analysis of how the company's media coverage aligns with their PR goals. Include strengths, weaknesses, and areas for improvement.",
            agent=pr_professional,
            context=[collect_articles_task]
        )
        
        write_report_task = Task(
            description="Write a comprehensive PR analysis report for {company} based on the analysis. Include an executive summary, key findings, and recommendations.",
            expected_output="A complete PR analysis report with executive summary, analysis of media coverage against PR goals, and actionable recommendations.",
            agent=report_writer,
            context=[analyze_articles_task]
        )
        
        # Create and return the crew
        return Crew(
            agents=[company_pr_analyst, pr_professional, report_writer],
            tasks=[collect_articles_task, analyze_articles_task, write_report_task],
            process=Process.sequential,
            memory=True,
            verbose=True
        )
