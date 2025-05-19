# preprocess.py
import pandas as pd

def load_bitext_dataset(csv_path):
    """
    Loads the Tech Support dataset and returns queries and structured metadata.
    """
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_cols = ["Customer_Issue", "Tech_Response", "Issue_Category", "Issue_Status"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    texts = df["Customer_Issue"].fillna("").tolist()
    metadatas = df[["Tech_Response", "Issue_Category", "Issue_Status"]].fillna("").rename(
        columns={
            "Tech_Response": "response",
            "Issue_Category": "category",
            "Issue_Status": "flags"
        }
    ).to_dict(orient="records")
    return texts, metadatas
