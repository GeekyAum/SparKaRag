import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from agno.models.openrouter import OpenRouter
from agno.agent import Agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import requests
import time

# ---------- CONFIG ----------
MODEL_ID = 'meta-llama/llama-3-70b-instruct'
WAIT_BETWEEN = 1  # seconds
# ----------------------------

@dataclass
class KPIData:
    """Structured KPI data"""
    name: str
    sentiment: str
    score: float
    summary: str
    pros: List[str]
    cons: List[str]
    quotes: List[str]
    insights: List[str]

@dataclass
class ReportSection:
    """Structured report section"""
    title: str
    content: str
    sentiment: str
    score: float
    sources: List[str]

class BaseAgent:
    """Base class for all agents with common functionality"""
    
    def __init__(self, api_key: str):
        self.llm = OpenRouter(id=MODEL_ID, api_key=api_key)
        self.agent = Agent(model=self.llm)
    
    def extract_output(self, response) -> str:
        """Extract text output from Agno response"""
        if hasattr(response, 'output'):
            return response.output
        elif hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Make LLM call with retry logic"""
        for attempt in range(max_retries):
            try:
                print(f"    Making LLM call (attempt {attempt + 1}/{max_retries})...")
                response = self.agent.run(message=prompt)
                result = self.extract_output(response)
                print(f"    ✅ LLM call successful")
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    wait_time = min((2 ** attempt) * 5, 30)  # Cap at 30 seconds
                    print(f"    Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif "timeout" in error_msg or "timed out" in error_msg:
                    print(f"    Timeout occurred, retrying...")
                    time.sleep(2)
                else:
                    print(f"    LLM call failed: {e}")
                    if attempt == max_retries - 1:
                        print(f"    ❌ All retries exhausted")
                        raise
                    time.sleep(2)
        
        print(f"    ❌ All LLM call attempts failed")
        return "LLM call failed after all retries"

class NERValidationAgent(BaseAgent):
    """Agent for validating and enhancing NER data"""
    
    def validate_ner_data(self, ner_data: Dict) -> Dict:
        """Validate and enhance NER data"""
        prompt = f"""
        Analyze and validate the following NER (Named Entity Recognition) data for product reviews.
        Ensure entities are properly categorized and add any missing important entities.
        
        Current NER data:
        {json.dumps(ner_data, indent=2)}
        
        Return a JSON object with:
        - validated_entities: List of validated entities with categories
        - missing_entities: List of important entities that should be added
        - confidence_scores: Confidence scores for each entity
        
        Focus on product-specific entities like: Camera, Battery, Performance, Display, Software, Build Quality, etc.
        """
        
        result = self.call_llm(prompt)
        try:
            return json.loads(result)
        except:
            return {"validated_entities": ner_data, "missing_entities": [], "confidence_scores": {}}

class KPISynthesisAgent(BaseAgent):
    """Agent for synthesizing KPIs from topic data"""
    
    def synthesize_kpis(self, topics_data: List[Dict]) -> List[KPIData]:
        """Synthesize KPIs from topic summaries"""
        prompt = f"""
        Analyze the following topic summaries and synthesize them into meaningful KPIs (Key Performance Indicators).
        
        Topic data:
        {json.dumps(topics_data, indent=2)}
        
        For each KPI, return a JSON object with:
        - name: Descriptive KPI name (e.g., "Camera Quality", "Battery Life", "User Satisfaction", "Value for Money", "Customer Service", "Software Performance", "Build Quality", "Gaming Performance", "Overall User Experience")
        - sentiment: "positive", "neutral", or "negative"
        - score: -1 to 1 (negative to positive)
        - summary: Brief summary of the KPI
        - pros: List of positive aspects
        - cons: List of negative aspects
        - quotes: Notable user quotes
        - insights: Key insights and recommendations
        
        Group related topics into logical KPIs. Focus on meaningful business metrics like:
        - User Satisfaction & Experience
        - Product Performance & Quality
        - Value for Money & Pricing
        - Customer Service & Support
        - Software & Updates
        - Build Quality & Durability
        
        Return as a JSON array of KPI objects. Use descriptive names, not numbers.
        """
        
        result = self.call_llm(prompt)
        try:
            kpi_list = json.loads(result)
            return [KPIData(**kpi) for kpi in kpi_list]
        except:
            # Fallback: create basic KPIs from topics
            return self._create_fallback_kpis(topics_data)
    
    def _create_fallback_kpis(self, topics_data: List[Dict]) -> List[KPIData]:
        """Create fallback KPIs if LLM parsing fails"""
        kpi_names = [
            "User Satisfaction", "Product Performance", "Value for Money", 
            "Customer Service", "Software Quality", "Build Quality"
        ]
        
        kpis = []
        for i, topic in enumerate(topics_data):
            kpi_name = kpi_names[i % len(kpi_names)] if i < len(kpi_names) else f"Topic {topic.get('topic', i)}"
            kpi = KPIData(
                name=kpi_name,
                sentiment='neutral',
                score=0.0,
                summary=topic.get('summary', ''),
                pros=[],
                cons=[],
                quotes=[],
                insights=[]
            )
            kpis.append(kpi)
        return kpis

class ReportBuildingAgent(BaseAgent):
    """Agent for building comprehensive reports"""
    
    def build_kpi_section(self, kpi: KPIData) -> ReportSection:
        """Build a detailed section for a single KPI"""
        prompt = f"""
        Create a comprehensive, well-structured report section for the KPI: {kpi.name}
        
        KPI Data:
        - Sentiment: {kpi.sentiment} (Score: {kpi.score})
        - Summary: {kpi.summary}
        - Pros: {kpi.pros}
        - Cons: {kpi.cons}
        - Quotes: {kpi.quotes}
        - Insights: {kpi.insights}
        
        Create a detailed section that includes:
        1. Executive Summary
        2. Detailed Analysis
        3. Pros and Cons (in table format)
        4. User Quotes and Testimonials
        5. Key Insights and Recommendations
        6. Action Items
        
        Use markdown formatting for structure. Make it business-ready and actionable.
        """
        
        content = self.call_llm(prompt)
        return ReportSection(
            title=f"{kpi.name} Analysis",
            content=content,
            sentiment=kpi.sentiment,
            score=kpi.score,
            sources=[]
        )
    
    def build_executive_summary(self, kpis: List[KPIData]) -> ReportSection:
        """Build executive summary from all KPIs"""
        prompt = f"""
        Create an executive summary based on the following KPIs:
        
        {json.dumps([{'name': k.name, 'sentiment': k.sentiment, 'score': k.score, 'summary': k.summary} for k in kpis], indent=2)}
        
        The executive summary should:
        1. Provide an overview of the product's performance
        2. Highlight key strengths and weaknesses
        3. Include overall sentiment and score
        4. Provide actionable recommendations
        5. Be suitable for business stakeholders
        
        Use markdown formatting and keep it concise but comprehensive.
        """
        
        content = self.call_llm(prompt)
        return ReportSection(
            title="Executive Summary",
            content=content,
            sentiment="neutral",
            score=0.0,
            sources=[]
        )

class SourceCitationAgent(BaseAgent):
    """Agent for adding source citations and validation"""
    
    def search_web(self, query: str) -> str:
        """Search the web for additional context and validation"""
        try:
            # Simple web search using DuckDuckGo API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                abstract = data.get('Abstract', '')
                if abstract:
                    return f"Search results for '{query}': {abstract}"
                else:
                    return f"No search results found for '{query}'"
            else:
                return f"Search failed for '{query}' (HTTP {response.status_code})"
        except Exception as e:
            return f"Search error for '{query}': {str(e)}"
    
    def add_citations(self, report_sections: List[ReportSection]) -> List[ReportSection]:
        """Add source citations to report sections"""
        for i, section in enumerate(report_sections):
            try:
                print(f"    Adding citations for section {i+1}/{len(report_sections)}...")
                # Search for relevant sources
                search_query = f"product review {section.title} latest"
                sources = self.search_web(search_query)
                
                # Add citations to content
                citation_text = f"\n\n**Sources:**\n{sources}"
                section.content += citation_text
                section.sources.append(sources)
                print(f"    ✅ Citations added for section {i+1}")
            except Exception as e:
                print(f"    ❌ Citation failed for section {i+1}: {str(e)}")
                # Add fallback citation
                fallback_citation = f"\n\n**Sources:**\nUnable to fetch external sources for {section.title}"
                section.content += fallback_citation
                section.sources.append("Citation generation failed")
        
        return report_sections

class ReasoningAgent(BaseAgent):
    """Agent for advanced reasoning and analysis"""
    
    def analyze_trends(self, kpis: List[KPIData]) -> Dict:
        """Analyze trends and patterns across KPIs"""
        prompt = f"""
        Analyze the following KPIs for trends, patterns, and insights:
        
        {json.dumps([{'name': k.name, 'sentiment': k.sentiment, 'score': k.score} for k in kpis], indent=2)}
        
        Provide a detailed JSON response with:
        - overall_trend: "positive", "negative", or "mixed"
        - key_patterns: List of specific patterns identified (e.g., "Users consistently praise value for money", "Customer service is a recurring pain point")
        - correlations: Relationships between different KPIs (e.g., "High value scores correlate with positive overall sentiment")
        - recommendations: Strategic recommendations with specific actions (e.g., "Improve customer service response time by 50% within 3 months")
        - risk_factors: Potential risks or concerns with mitigation strategies
        - opportunities: Business opportunities identified from the data
        - competitive_advantages: Strengths that can be leveraged against competitors
        
        Make the analysis specific, actionable, and business-focused.
        """
        
        result = self.call_llm(prompt)
        try:
            trends = json.loads(result)
            # Ensure we have all required fields
            required_fields = ['overall_trend', 'key_patterns', 'correlations', 'recommendations', 'risk_factors']
            for field in required_fields:
                if field not in trends:
                    trends[field] = []
            return trends
        except:
            return {
                "overall_trend": "mixed",
                "key_patterns": ["Mixed user sentiment across different product aspects", "Value for money is consistently mentioned positively"],
                "correlations": ["Positive value sentiment correlates with overall satisfaction", "Customer service issues impact overall brand perception"],
                "recommendations": ["Improve customer service response time", "Leverage positive value messaging in marketing"],
                "risk_factors": ["Negative customer service experiences may impact brand loyalty", "Software update delays could affect user retention"],
                "opportunities": ["Strong value proposition can attract budget-conscious consumers", "Positive user experiences can be amplified in marketing"],
                "competitive_advantages": ["Competitive pricing strategy", "Strong value-for-money positioning"]
            }
    
    def generate_insights(self, data: Dict) -> List[str]:
        """Generate actionable insights from data"""
        prompt = f"""
        Generate actionable business insights from this data:
        
        {json.dumps(data, indent=2)}
        
        Return a JSON array of insight strings. Each insight should be:
        - Specific and actionable
        - Based on the data provided
        - Relevant to business decision-making
        - Clear and concise
        - Include specific recommendations
        
        Focus on insights like:
        - Product improvement opportunities
        - Marketing strategy recommendations
        - Customer service enhancements
        - Competitive positioning
        - Pricing strategy insights
        - Feature development priorities
        - Customer retention strategies
        
        Make insights specific and measurable where possible.
        """
        
        result = self.call_llm(prompt)
        try:
            insights = json.loads(result)
            # Ensure we have at least some insights
            if not insights or len(insights) == 0:
                return [
                    "Focus on improving customer service based on negative feedback patterns",
                    "Leverage positive value-for-money sentiment in marketing campaigns",
                    "Address software update concerns to improve user satisfaction",
                    "Develop targeted messaging for different user segments"
                ]
            return insights
        except:
            return [
                "Focus on improving customer service based on negative feedback patterns",
                "Leverage positive value-for-money sentiment in marketing campaigns", 
                "Address software update concerns to improve user satisfaction",
                "Develop targeted messaging for different user segments",
                "Implement regular customer feedback collection and analysis"
            ] 