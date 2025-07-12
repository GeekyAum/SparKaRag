import json
import os
from typing import Dict, List, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents import (
    NERValidationAgent, KPISynthesisAgent, ReportBuildingAgent, 
    SourceCitationAgent, ReasoningAgent, KPIData, ReportSection
)

# ---------- STATE DEFINITION ----------
class WorkflowState(TypedDict):
    """State for the LangGraph workflow"""
    api_key: Annotated[str, {"allow_multiple": True}]
    metadata: Dict
    ner_data: Dict
    validated_ner: Dict
    topics_data: List[Dict]
    kpis: List[KPIData]
    report_sections: List[ReportSection]
    executive_summary: ReportSection
    trends_analysis: Dict
    insights: List[str]
    final_report: str
    errors: List[str]

# ---------- WORKFLOW NODES ----------
def validate_ner(state: WorkflowState) -> WorkflowState:
    """Node: Validate and enhance NER data"""
    try:
        print("🔍 Validating NER data...")
        agent = NERValidationAgent(state["api_key"])
        validated_ner = agent.validate_ner_data(state["ner_data"])
        state["validated_ner"] = validated_ner
        print("✅ NER validation completed")
    except Exception as e:
        state["errors"].append(f"NER validation failed: {str(e)}")
        state["validated_ner"] = state["ner_data"]  # Fallback
    return state

def synthesize_kpis(state: WorkflowState) -> WorkflowState:
    """Node: Synthesize KPIs from topic data"""
    try:
        print("📊 Synthesizing KPIs...")
        agent = KPISynthesisAgent(state["api_key"])
        
        # Limit to top 10 topics for faster processing
        limited_topics = state["topics_data"][:10] if len(state["topics_data"]) > 10 else state["topics_data"]
        print(f"Processing {len(limited_topics)} topics (limited from {len(state['topics_data'])})")
        
        kpis = agent.synthesize_kpis(limited_topics)
        state["kpis"] = kpis
        print(f"✅ Synthesized {len(kpis)} KPIs")
    except Exception as e:
        state["errors"].append(f"KPI synthesis failed: {str(e)}")
        state["kpis"] = []
    return state

def build_report_sections(state: WorkflowState) -> WorkflowState:
    """Node: Build detailed report sections for each KPI"""
    try:
        print("📝 Building report sections...")
        agent = ReportBuildingAgent(state["api_key"])
        sections = []
        
        # Limit to top 5 KPIs for faster processing
        limited_kpis = state["kpis"][:5] if len(state["kpis"]) > 5 else state["kpis"]
        print(f"Building sections for {len(limited_kpis)} KPIs (limited from {len(state['kpis'])})")
        
        for i, kpi in enumerate(limited_kpis):
            print(f"  - Building section {i+1}/{len(limited_kpis)} for {kpi.name}...")
            try:
                section = agent.build_kpi_section(kpi)
                sections.append(section)
                print(f"    ✅ Section {i+1} completed")
            except Exception as e:
                print(f"    ❌ Section {i+1} failed: {str(e)}")
                # Create a fallback section
                fallback_section = ReportSection(
                    title=f"{kpi.name} Analysis",
                    content=f"Analysis for {kpi.name} could not be generated due to an error.",
                    sentiment=kpi.sentiment,
                    score=kpi.score,
                    sources=[]
                )
                sections.append(fallback_section)
        
        state["report_sections"] = sections
        print(f"✅ Built {len(sections)} report sections")
    except Exception as e:
        state["errors"].append(f"Report building failed: {str(e)}")
        state["report_sections"] = []
    return state

def build_executive_summary(state: WorkflowState) -> WorkflowState:
    """Node: Build executive summary"""
    try:
        print("📋 Building executive summary...")
        agent = ReportBuildingAgent(state["api_key"])
        summary = agent.build_executive_summary(state["kpis"])
        state["executive_summary"] = summary
        print("✅ Executive summary completed")
    except Exception as e:
        state["errors"].append(f"Executive summary failed: {str(e)}")
        # Create fallback summary
        state["executive_summary"] = ReportSection(
            title="Executive Summary",
            content="Executive summary could not be generated due to an error.",
            sentiment="neutral",
            score=0.0,
            sources=[]
        )
    return state

def analyze_trends(state: WorkflowState) -> WorkflowState:
    """Node: Analyze trends and patterns"""
    try:
        print("📈 Analyzing trends...")
        agent = ReasoningAgent(state["api_key"])
        trends = agent.analyze_trends(state["kpis"])
        state["trends_analysis"] = trends
        print("✅ Trends analysis completed")
    except Exception as e:
        state["errors"].append(f"Trends analysis failed: {str(e)}")
        state["trends_analysis"] = {
            "overall_trend": "mixed",
            "key_patterns": [],
            "correlations": [],
            "recommendations": [],
            "risk_factors": []
        }
    return state

def add_citations(state: WorkflowState) -> WorkflowState:
    """Node: Add source citations"""
    try:
        print("🔗 Adding citations...")
        agent = SourceCitationAgent(state["api_key"])
        sections_with_citations = agent.add_citations(state["report_sections"])
        state["report_sections"] = sections_with_citations
        print("✅ Citations added")
    except Exception as e:
        state["errors"].append(f"Citation addition failed: {str(e)}")
    return state

def generate_insights(state: WorkflowState) -> WorkflowState:
    """Node: Generate actionable insights"""
    try:
        print("💡 Generating insights...")
        agent = ReasoningAgent(state["api_key"])
        
        # Combine all data for insights
        insight_data = {
            "kpis": [{"name": k.name, "sentiment": k.sentiment, "score": k.score} for k in state["kpis"]],
            "trends": state["trends_analysis"],
            "metadata": state["metadata"]
        }
        
        insights = agent.generate_insights(insight_data)
        state["insights"] = insights
        print(f"✅ Generated {len(insights)} insights")
    except Exception as e:
        state["errors"].append(f"Insight generation failed: {str(e)}")
        state["insights"] = []
    return state

def compile_final_report(state: WorkflowState) -> WorkflowState:
    """Node: Compile final report"""
    try:
        print("📄 Compiling final report...")
        
        # Build the complete report
        report_parts = []
        
        # Title
        report_parts.append("# Product Analysis Report\n")
        
        # Executive Summary
        if state["executive_summary"]:
            report_parts.append(f"## {state['executive_summary'].title}\n")
            report_parts.append(state["executive_summary"].content)
            report_parts.append("\n---\n")
        
        # KPI Sections
        for section in state["report_sections"]:
            report_parts.append(f"## {section.title}\n")
            report_parts.append(section.content)
            report_parts.append("\n---\n")
        
        # Trends Analysis
        if state["trends_analysis"]:
            report_parts.append("## Trends and Patterns Analysis\n")
            trends = state["trends_analysis"]
            report_parts.append(f"**Overall Trend:** {trends.get('overall_trend', 'Unknown')}\n")
            
            if trends.get('key_patterns'):
                report_parts.append("**Key Patterns:**\n")
                for pattern in trends['key_patterns']:
                    report_parts.append(f"- {pattern}\n")
            
            if trends.get('correlations'):
                report_parts.append("**Correlations:**\n")
                for correlation in trends['correlations']:
                    report_parts.append(f"- {correlation}\n")
            
            if trends.get('recommendations'):
                report_parts.append("**Strategic Recommendations:**\n")
                for rec in trends['recommendations']:
                    report_parts.append(f"- {rec}\n")
            
            if trends.get('opportunities'):
                report_parts.append("**Business Opportunities:**\n")
                for opp in trends['opportunities']:
                    report_parts.append(f"- {opp}\n")
            
            if trends.get('competitive_advantages'):
                report_parts.append("**Competitive Advantages:**\n")
                for adv in trends['competitive_advantages']:
                    report_parts.append(f"- {adv}\n")
            
            if trends.get('risk_factors'):
                report_parts.append("**Risk Factors:**\n")
                for risk in trends['risk_factors']:
                    report_parts.append(f"- {risk}\n")
            
            report_parts.append("\n---\n")
        
        # Insights
        if state["insights"]:
            report_parts.append("## Key Insights and Action Items\n")
            for i, insight in enumerate(state["insights"], 1):
                report_parts.append(f"{i}. {insight}\n")
            report_parts.append("\n---\n")
        
        # Errors (if any)
        if state["errors"]:
            report_parts.append("## Processing Notes\n")
            report_parts.append("The following issues were encountered during report generation:\n")
            for error in state["errors"]:
                report_parts.append(f"- {error}\n")
        
        # Compile final report
        final_report = "\n".join(report_parts)
        state["final_report"] = final_report
        
        # Save to file
        with open("product_report.txt", "w", encoding="utf-8") as f:
            f.write(final_report)
        
        print("✅ Final report compiled and saved")
        
    except Exception as e:
        state["errors"].append(f"Report compilation failed: {str(e)}")
        state["final_report"] = "Report compilation failed due to an error."
    
    return state

# ---------- WORKFLOW DEFINITION ----------
def create_workflow() -> StateGraph:
    """Create the LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("validate_ner", validate_ner)
    workflow.add_node("synthesize_kpis", synthesize_kpis)
    workflow.add_node("build_sections", build_report_sections)
    workflow.add_node("build_summary", build_executive_summary)
    workflow.add_node("analyze_trends", analyze_trends)
    workflow.add_node("add_citations", add_citations)
    workflow.add_node("generate_insights", generate_insights)
    workflow.add_node("compile_report", compile_final_report)
    
    # Define the flow
    workflow.set_entry_point("validate_ner")
    workflow.add_edge("validate_ner", "synthesize_kpis")
    workflow.add_edge("synthesize_kpis", "build_summary")
    workflow.add_edge("build_summary", "build_sections")
    workflow.add_edge("build_sections", "analyze_trends")
    workflow.add_edge("analyze_trends", "add_citations")
    workflow.add_edge("add_citations", "generate_insights")
    workflow.add_edge("generate_insights", "compile_report")
    workflow.add_edge("compile_report", END)
    
    return workflow

# ---------- WORKFLOW EXECUTION ----------
def run_workflow(api_key: str, metadata_file: str = "metadata.json", ner_file: str = "RAG/extra_context/context_store.json") -> Dict:
    """Run the complete workflow"""
    
    # Load data
    def load_json(path):
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Load metadata and handle different structures
    metadata_raw = load_json(metadata_file)
    if isinstance(metadata_raw, list):
        # If it's a list, treat it as topics directly
        topics_data = metadata_raw
        metadata = {"topics": topics_data}
    else:
        # If it's a dict, extract topics
        topics_data = metadata_raw.get("topics", [])
        metadata = metadata_raw
    
    # Initialize state
    initial_state = WorkflowState(
        api_key=api_key,
        metadata=metadata,
        ner_data=load_json(ner_file),
        validated_ner={},
        topics_data=topics_data,
        kpis=[],
        report_sections=[],
        executive_summary=None,
        trends_analysis={},
        insights=[],
        final_report="",
        errors=[]
    )
    
    # Create and run workflow
    workflow = create_workflow()
    app = workflow.compile(checkpointer=None)
    
    print("🚀 Starting advanced report generation workflow...")
    print("=" * 60)
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    print("=" * 60)
    print("🎉 Workflow completed!")
    
    if result["errors"]:
        print(f"⚠️  {len(result['errors'])} errors encountered:")
        for error in result["errors"]:
            print(f"   - {error}")
    
    return result 