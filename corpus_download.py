from huggingface_hub import snapshot_download
import os
import sys

# Get the absolute path to the project root directory (where the script is located)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir  # Changed from os.path.dirname(script_dir)

# Ensure the output directory exists
output_dir = os.path.join(project_root, "corpus", "fineweb")
os.makedirs(output_dir, exist_ok=True)

try:
    print(f"Starting download to: {output_dir}")
    folder = snapshot_download(
        "HuggingFaceFW/fineweb", 
        repo_type="dataset",
        local_dir=output_dir,
        # replace "data/CC-MAIN-2023-50/*" with "sample/100BT/*" to use the 100BT sample
        allow_patterns="sample/10BT/*",
        max_workers=4,  # Limit concurrent downloads
        force_download=False  # Allow resuming interrupted downloads
    )
    print("Download completed successfully!")
except Exception as e:
    print(f"Error during download: {str(e)}", file=sys.stderr)
    sys.exit(1)