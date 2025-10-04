import pandas as pd

def load_and_preprocess(file_path="dataset.csv"):
    # Load dataset
    df = pd.read_csv(file_path)

    # Drop duplicates (if any)
    df.drop_duplicates(inplace=True)

    # Handle missing values: fill with mean or drop (choose wisely)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Select useful features for ML
    features = [
        "teff", "logg", "feh", "radius", "mass", "dens", "kepmag"
    ]
    target = "nconfp"  # Number of confirmed planets (can be binary label)

    # Keep only selected columns if present
    available_features = [f for f in features if f in df.columns]
    if target in df.columns:
        df = df[available_features + [target]]
    else:
        df = df[available_features]

    return df
