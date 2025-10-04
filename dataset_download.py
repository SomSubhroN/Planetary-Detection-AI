import kagglehub

# Download latest version
path = kagglehub.dataset_download("vijayveersingh/kepler-and-tess-exoplanet-data")

print("Path to dataset files:", path)