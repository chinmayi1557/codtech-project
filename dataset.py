import kagglehub

# Download latest version
path = kagglehub.dataset_download("ak0212/anxiety-and-depression-mental-health-factors")

print("Path to dataset files:", path)
print(path.head())