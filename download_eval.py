import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

def main():
    # Load credentials from .env
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("Error: HF_TOKEN not found in .env file.")
        return
        
    repo_id = os.environ.get("HF_REPO_ID", "Avi2006/spatial-moe-results")
    
    print(f"Fetching 'evaluation_results' from {repo_id}...")
    
    try:
        # Download only the evaluation_results folder directly to the current directory
        # snapshot_download preserves the repo structure, so it will create ./evaluation_results locally
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns="evaluation_results/*",
            local_dir=".",
            token=hf_token
        )
        print(f"\nSuccess! The evaluation_results folder has been downloaded to:")
        print(os.path.join(os.path.abspath("."), "evaluation_results"))
    except Exception as e:
        print(f"Failed to download evaluation_results: {e}")

if __name__ == "__main__":
    main()
