import pandas as pd
import glob
import os

DATA_DIR = r"C:\Users\kerne\ir-assignment-2\data"
OUT_FILE = os.path.join(DATA_DIR, "all_movies.csv")

def merge_csv_files():
    # Searching for ANY csv with "movies" in the filename
    pattern = os.path.join(DATA_DIR, "*movies*.csv")
    files = glob.glob(pattern)

    print("Found CSV files:", files)

    if len(files) == 0:
        print("ERROR: No CSV files found. Check DATA_DIR path.")
        return

    dfs = []
    for f in files:
        print("Reading:", f)
        df = pd.read_csv(f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(OUT_FILE, index=False)
    print("Saved merged dataset to:", OUT_FILE)

if __name__ == "__main__":
    merge_csv_files()

