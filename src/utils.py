import pandas as pd
import os


def load_dataset(path):
    """
    Safely load a dataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def categorize_risk(score):
    """
    Convert numerical risk score into category
    """
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Medium"
    else:
        return "High"


def extract_time_features(df, column):
    """
    Extract date and hour from datetime column
    """
    df[column] = pd.to_datetime(df[column])
    df["date"] = df[column].dt.date
    df["hour"] = df[column].dt.hour
    return df
