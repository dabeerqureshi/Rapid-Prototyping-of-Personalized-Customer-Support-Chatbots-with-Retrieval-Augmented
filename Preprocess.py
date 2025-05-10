import pandas as pd

def load_bitext_dataset(csv_path):
    """
    Loads the Bitext Customer Support dataset and returns the instruction texts and associated metadata.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        tuple: A tuple containing:
            - texts (list): List of instruction strings.
            - metadatas (list): List of dictionaries with response, intent, category, and flags.
    """
    df = pd.read_csv(csv_path)
    texts = df["instruction"].tolist()
    metadatas = df[["response", "intent", "category", "flags"]].to_dict(orient="records")
    return texts, metadatas
