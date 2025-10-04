import os
import glob
import pandas as pd

# Path where CSV files are stored
DATASET_PATH = "datasets/"
OUTPUT_FILE = "dataset.csv"

def load_and_merge_datasets(path=DATASET_PATH):
    # Collect all CSV files in datasets/
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    print(f"Found {len(csv_files)} CSV files.")

    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            print(f"Loaded {file} with shape {df.shape}")
            dataframes.append(df)
        except Exception as e:
            print(f"❌ Could not load {file}: {e}")

    # Merge datasets (only common columns kept)
    merged_df = pd.concat(dataframes, axis=0, ignore_index=True, sort=False)

    print(f"Total merged shape: {merged_df.shape}")
    return merged_df


def preprocess_data(df):
    # Drop duplicate rows
    df = df.drop_duplicates()

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Fill missing values with median (numeric) or mode (categorical)
    for col in df.columns:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def save_dataset(df, filename=OUTPUT_FILE):
    df.to_csv(filename, index=False)
    print(f"✅ Processed dataset saved to {filename}")


if __name__ == "__main__":
    merged = load_and_merge_datasets()
    processed = preprocess_data(merged)
    save_dataset(processed)
