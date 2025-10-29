from typing import List, Dict
from load_dotenv import load_dotenv
from pydantic import BaseModel, Field
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
model="gpt-4.1"
load_dotenv()
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ResearchTask(BaseModel):
    analysis_type: str = Field(description="Type of market analysis to conduct")
    research_focus: str = Field(description="What this analysis should investigate")
    methodology: str = Field(description="Research approach for this section")
    depth_level: str = Field(description="Level of detail required (high/medium/low)")


class OrchestratorPlan(BaseModel):
    market_overview: str = Field(description="High-level market context and scope")
    research_objectives: List[str] = Field(description="Key questions to answer")
    target_segments: List[str] = Field(description="Market segments to analyze")
    analysis_sections: List[ResearchTask] = Field(description="List of research tasks")


class ResearchFindings(BaseModel):
    """Findings from a market research worker"""
    key_insights: List[str] = Field(description="Primary insights discovered")
    data_points: List[str] = Field(description="Important metrics and statistics")
    analysis_content: str = Field(description="Detailed analysis content")
    recommendations: List[str] = Field(description="Actionable recommendations")


class SectionRecommendations(BaseModel):
    """Recommended improvements for a research section"""
    section_name: str = Field(description="Name of the research section")
    improvement_suggestion: str = Field(description="Suggested improvement")
    priority: str = Field(description="Priority level: high/medium/low")


class FinalReview(BaseModel):
    """Final review and consolidated report"""
    analytical_rigor_score: float = Field(description="Quality of analysis (0-1)")
    insight_coherence_score: float = Field(description="How well insights connect (0-1)")
    section_improvements: List[SectionRecommendations] = Field(
        description="Suggested improvements by section"
    )
    executive_summary: str = Field(description="High-level summary of key findings")
    final_report: str = Field(description="Complete, polished market research report")



ORCHESTRATOR_PROMPT = """
Design a comprehensive market research plan for this request:

Market/Industry: {market}
Research Scope: {scope}
Business Context: {context}
Timeline: {timeline}

Structure your analysis plan to cover:

# Market Overview
Provide context about the market landscape and define the research boundaries.

# Research Objectives  
List 3-5 key questions this research should answer.

# Target Segments
Identify market segments that should be analyzed separately.

# Analysis Sections
Break down into specific research tasks:
## Section 1: [Analysis Type]
- Research Focus: what to investigate
- Methodology: research approach  
- Depth Level: high/medium/low

[Additional sections as needed - typically 4-6 sections covering competitive landscape, market sizing, trends, customer analysis, etc.]
"""

WORKER_PROMPT = """
Conduct market research analysis based on:

Market: {market}
Analysis Type: {analysis_type}
Research Focus: {research_focus}
Methodology: {methodology}
Depth Level: {depth_level}

Previous Research Context:
{previous_findings}

Structure your response as:

# Key Insights
- Primary insight 1
- Primary insight 2
[Additional key findings...]

# Data Points
- Important metric 1
- Market statistic 2
[Additional quantitative findings...]

# Analysis Content
[Detailed analysis following the specified methodology and depth level]

# Recommendations
- Actionable recommendation 1
- Strategic suggestion 2
[Additional recommendations...]
"""

REVIEWER_PROMPT = """
Review this market research report for analytical quality and coherence:

Market: {market}
Research Objectives: {objectives}

Research Sections:
{sections}

Evaluate the report on:
1. Analytical rigor (methodology, data quality, logical reasoning)
2. Insight coherence (how well findings connect across sections)
3. Actionability of recommendations
4. Overall research quality

Provide scores between 0.0 and 1.0 for analytical rigor and insight coherence.
Suggest specific improvements for each section if needed.
Create an executive summary highlighting the most important findings.
Produce a final polished report that integrates all sections coherently.
"""

# implement orchestrator
class MarketResearchOrchestrator:
    def __init__(self):
        self.research_findings = {}

    def create_research_plan(self, market: str, scope: str, context: str, timeline: str) -> OrchestratorPlan:
        """Get orchestrator's market research plan"""
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": ORCHESTRATOR_PROMPT.format(
                        market=market, scope=scope, context=context, timeline=timeline
                    ),
                }
            ],
            response_format=OrchestratorPlan,
        )
        return completion.choices[0].message.parsed

    def conduct_analysis(self, market: str, task: ResearchTask) -> ResearchFindings:
        """Worker: Conduct specific market research analysis with context from previous findings."""
        # Create context from previously completed research
        previous_context = "\n\n".join(
            [
                f"=== {analysis_type} ===\nKey Insights: {findings.key_insights}\nData: {findings.data_points}"
                for analysis_type, findings in self.research_findings.items()
            ]
        )

        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": WORKER_PROMPT.format(
                        market=market,
                        analysis_type=task.analysis_type,
                        research_focus=task.research_focus,
                        methodology=task.methodology,
                        depth_level=task.depth_level,
                        previous_findings=previous_context
                        if previous_context
                        else "This is the first analysis section.",
                    ),
                }
            ],
            response_format=ResearchFindings,
        )
        return completion.choices[0].message.parsed

    def review_report(self, market: str, plan: OrchestratorPlan) -> FinalReview:
        """Reviewer: Analyze research quality and create final report"""
        sections_text = "\n\n".join(
            [
                f"=== {analysis_type} ===\nInsights: {findings.key_insights}\nAnalysis: {findings.analysis_content}\nRecommendations: {findings.recommendations}"
                for analysis_type, findings in self.research_findings.items()
            ]
        )

        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": REVIEWER_PROMPT.format(
                        market=market,
                        objectives=plan.research_objectives,
                        sections=sections_text,
                    ),
                }
            ],
            response_format=FinalReview,
        )
        return completion.choices[0].message.parsed

    def generate_market_research(
        self,
        market: str,
        scope: str = "comprehensive analysis",
        context: str = "strategic planning",
        timeline: str = "Q1 2025"
    ) -> Dict:
        """Process the entire market research task"""
        logger.info(f"Starting market research for: {market}")

        # Create research plan
        plan = self.create_research_plan(market, scope, context, timeline)
        logger.info(f"Research plan created: {len(plan.analysis_sections)} analysis sections")

        # Conduct each analysis section
        for task in plan.analysis_sections:
            logger.info(f"Conducting analysis: {task.analysis_type}")
            findings = self.conduct_analysis(market, task)
            self.research_findings[task.analysis_type] = findings

        # Review and synthesize final report
        logger.info("Reviewing and synthesizing final report")
        final_review = self.review_report(market, plan)

        return {
            "research_plan": plan,
            "findings": self.research_findings,
            "final_review": final_review
        }


if __name__ == "__main__":
    orchestrator = MarketResearchOrchestrator()

    # Example: Electric vehicle charging infrastructure market
    market = "Electric Vehicle Charging Infrastructure"
    result = orchestrator.generate_market_research(
        market=market,
        scope="North American market analysis",
        context="Investment opportunity assessment",
        timeline="2025-2027"
    )

    print("\n" + "=" * 60)
    print("EXECUTIVE SUMMARY")
    print("=" * 60)
    print(result["final_review"].executive_summary)

    print("\n" + "=" * 60)
    print("COMPLETE MARKET RESEARCH REPORT")
    print("=" * 60)
    print(result["final_review"].final_report)

    print(f"\nAnalytical Rigor Score: {result['final_review'].analytical_rigor_score}")
    print(f"Insight Coherence Score: {result['final_review'].insight_coherence_score}")

    if result["final_review"].section_improvements:
        print("\nSuggested Improvements:")
        for improvement in result["final_review"].section_improvements:
            print(f"- {improvement.section_name} ({improvement.priority}): {improvement.improvement_suggestion}")

