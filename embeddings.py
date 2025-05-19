import os
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


def upload_to_pinecone(
    texts,
    metadatas,
    index_name="tech-support-chatbot-customer-support",
    pinecone_api_key=None,
    model_name="all-MiniLM-L6-v2",
    batch_size=100,
    cloud="aws",
    region="us-east-1"
):
    """
    Uploads sentence embeddings with metadata to Pinecone in batches.

    Args:
        texts (list): List of input text strings.
        metadatas (list): List of dicts with metadata corresponding to each text.
        index_name (str): Pinecone index name to use or create.
        pinecone_api_key (str): Your Pinecone API key.
        model_name (str): SentenceTransformer model name.
        batch_size (int): Batch size for uploading to Pinecone.
        cloud (str): Cloud provider for Pinecone.
        region (str): Cloud region for Pinecone.
    """
    if not pinecone_api_key:
        raise ValueError("Pinecone API key must be provided.")

    print("🔗 Loading SentenceTransformer model...")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()

    os.environ["PINECONE_API_KEY"] = "pcsk_33qtrC_5gTrvztaWPk6Kzz4m6ZDa4vvJPeGuhWS5wRtPDow6MvMoCo7pHwsKxAHrzcFsay"
    pc = Pinecone(api_key=pinecone_api_key)

    # Create index if not exists
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
        print("Waiting for index to initialize...")
        time.sleep(10)
    else:
        print(f"Using existing Pinecone index: {index_name}")

    index = pc.Index(index_name)

    # Upload in batches
    print("🚀 Uploading to Pinecone...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        batch_ids = [f"bitext-{i + j}" for j in range(len(batch_texts))]
        batch_embeddings = model.encode(batch_texts).tolist()

        to_upsert = [
            {"id": batch_ids[j], "values": batch_embeddings[j], "metadata": batch_metas[j]}
            for j in range(len(batch_texts))
        ]
        index.upsert(to_upsert)

    print("✅ Done uploading dataset to Pinecone!")
