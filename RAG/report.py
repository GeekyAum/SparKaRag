import os
from getpass import getpass
from workflow import run_workflow

# ---------- CONFIG ----------
METADATA_FILE = 'metadata.json'
REPORT_FILE = 'product_report.txt'
NER_CONTEXT_FILE = 'RAG/extra_context/context_store.json'
# ----------------------------

def main():
    """Main function to run the advanced report generation workflow"""
    
    print("🤖 Advanced Product Analysis Report Generator")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("🔑 Please enter your OpenRouter API key:")
        api_key = getpass("API Key: ").strip()
        if not api_key:
            print("❌ API key is required!")
            return
    
    # Check if required files exist
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Required file not found: {METADATA_FILE}")
        print("Please run the previous pipeline steps first (topical_chunking.py, summarization.py)")
        return
    
    if not os.path.exists(NER_CONTEXT_FILE):
        print(f"❌ Required file not found: {NER_CONTEXT_FILE}")
        print("Please run the previous pipeline steps first (inject_context.py)")
        return
    
    print("✅ All required files found")
    print("🚀 Starting advanced report generation with agent team...")
    print()
    
    try:
        # Run the workflow
        result = run_workflow(api_key, METADATA_FILE, NER_CONTEXT_FILE)
        
        print()
        print("📊 Report Generation Summary:")
        print(f"   • KPIs synthesized: {len(result.get('kpis', []))}")
        print(f"   • Report sections: {len(result.get('report_sections', []))}")
        print(f"   • Insights generated: {len(result.get('insights', []))}")
        print(f"   • Errors encountered: {len(result.get('errors', []))}")
        
        if result.get('final_report'):
            print(f"   • Report saved to: {REPORT_FILE}")
            print()
            print("📄 Report Preview (first 500 characters):")
            print("-" * 50)
            print(result['final_report'])
        
    except Exception as e:
        print(f"❌ Workflow failed: {str(e)}")
        print("Please check your API key and try again.")

if __name__ == "__main__":
    main()
