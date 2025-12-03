import os
import pandas as pd

DATA_PATH = "data/processed"

def load_movielens_processed(dataset):
    # movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
    return pd.read_csv(os.path.join(DATA_PATH, dataset))
    # return pd.read_csv("data/processed/movielens_merged.csv")
