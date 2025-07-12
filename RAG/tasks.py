import subprocess
import sys
import os
import time
from getpass import getpass

def run_script(script_name, args=None):
    """Run a Python script with optional arguments"""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run without capturing output to see real-time progress
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        print(f"❌ Script {script_name} failed with return code {result.returncode}")
        return False
    return True

def main():
    print("🚀 RAG Pipeline with Advanced Agent Team")
    print("=" * 50)
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("🔑 Please enter your OpenRouter API key:")
        api_key = getpass("API Key: ").strip()
        if not api_key:
            print("❌ API key is required!")
            return
        os.environ['OPENROUTER_API_KEY'] = api_key
    
    input_file = "motorolaedge50fusion_comments.txt"
    
    print(f"📁 Input file: {input_file}")
    print(f"🔑 API Key: {api_key[:10]}...{api_key[-4:]}")
    print()
    
    # Step 1: Topical Chunking
    print("🔹 Step 1: Topical Chunking")
    if not run_script("topical_chunking.py", ['--input', input_file]):
        print("❌ Topical chunking failed!")
        return
    
    # Step 2: Injecting NER + Sentiment Context
    print("\n🔹 Step 2: Injecting NER + Sentiment Context")
    ner_json_path = "RAG/extra_context/context_store.json"
    if not run_script("inject_context.py", ['--ner', ner_json_path]):
        print("❌ Context injection failed!")
        return
    
    # Step 3: Summarization + Indexing
    print("\n🔹 Step 3: Summarization + Indexing")
    if not run_script("summarization.py"):
        print("❌ Summarization failed!")
        return
    
    # Step 4: Advanced Report Generation with Agent Team
    print("\n🔹 Step 4: Advanced Report Generation with Agent Team")
    if not run_script("report.py"):
        print("❌ Report generation failed!")
        return
    
    # Step 5: Starting Query Mode
    print("\n🔹 Step 5: Starting Query Mode")
    if not run_script("query.py"):
        print("❌ Query mode failed!")
        return
    
    print("\n🎉 Pipeline completed successfully!")
    print("📄 Check 'product_report.txt' for the advanced report")
    print("💬 Use query.py for interactive Q&A")

if __name__ == "__main__":
    main()
