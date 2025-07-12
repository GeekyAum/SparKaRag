import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ner', required=True, help='Path to NER+Sentiment JSON file')
    args = parser.parse_args()

    # Load the provided JSON
    try:
        with open(args.ner, 'r') as f:
            ner_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load NER JSON: {e}")
        exit(1)

    # Ensure output folder exists
    os.makedirs("RAG/extra_context", exist_ok=True)

    # Save it to a known shared path
    output_path = "RAG/extra_context/context_store.json"
    try:
        with open(output_path, 'w') as f:
            json.dump(ner_data, f, indent=2)
        print(f"✅ NER + Sentiment context saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save context: {e}")
        exit(1)

if __name__ == "__main__":
    main()
