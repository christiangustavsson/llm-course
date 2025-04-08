from huggingface_hub import snapshot_download
import os

# Get the absolute path to the project root directory (one level up from the script)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Ensure the output directory exists
output_dir = os.path.join(project_root, "datasets", "fineweb")
os.makedirs(output_dir, exist_ok=True)

folder = snapshot_download(
                "HuggingFaceFW/fineweb", 
                repo_type="dataset",
                local_dir=output_dir,
                # replace "data/CC-MAIN-2023-50/*" with "sample/100BT/*" to use the 100BT sample
                allow_patterns="sample/10BT/*")