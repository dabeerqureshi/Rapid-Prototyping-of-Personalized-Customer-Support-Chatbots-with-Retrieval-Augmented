# main.py

from Preprocess import load_bitext_dataset
from embeddings import upload_to_pinecone
from rag_query import rag_query
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone

# Paths and configuration
csv_path = r"C:\Users\qures\OneDrive\Desktop\Generative AI Project\Dataset\Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
pinecone_api_key = "pcsk_33qtrC_5gTrvztaWPk6Kzz4m6ZDa4vvJPeGuhWS5wRtPDow6MvMoCo7pHwsKxAHrzcFsay"  # Replace with your actual Pinecone API key

# Load dataset
texts, metadatas = load_bitext_dataset(csv_path)

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "chatbot-customer-support1"
index = pc.Index(index_name)

# Check if the Pinecone index already exists
existing_indexes = [idx["name"] for idx in pc.list_indexes()]
if index_name not in existing_indexes:
    # Upload data to Pinecone if the index does not exist
    upload_to_pinecone(
        texts=texts,
        metadatas=metadatas,
        index_name=index_name,
        pinecone_api_key=pinecone_api_key,
        model_name="all-MiniLM-L6-v2",
        batch_size=100
    )
else:
    print(f"Index '{index_name}' already exists. Skipping upload.")

# Initialize LLM and SentenceTransformer model
llm = pipeline("text2text-generation", model="google/flan-t5-base")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test RAG query
query = "What is your return policy?"
answer = rag_query(query, model=model, index=index, llm=llm)
print(f"💬 Q: {query}\n🧠 A: {answer}")
