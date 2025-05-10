# ask_question.py

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone
from rag_query import rag_query

# Configuration
pinecone_api_key = "pcsk_33qtrC_5gTrvztaWPk6Kzz4m6ZDa4vvJPeGuhWS5wRtPDow6MvMoCo7pHwsKxAHrzcFsay"
index_name = "chatbot-customer-support1"

# Initialize Pinecone and index
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Load models
llm = pipeline("text2text-generation", model="google/flan-t5-base")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ask a question
query = input("❓ Ask your question: ")
answer = rag_query(query, model=model, index=index, llm=llm)
print(f"\n💬 Q: {query}\n🧠 A: {answer}")
