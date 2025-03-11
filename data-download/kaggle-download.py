import kagglehub

# Download latest version
path = kagglehub.dataset_download("datasets/openwebtext/")

print("Path to dataset files:", path)