# test_bot.py

import time
import json
import csv
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone
from numpy import dot
from numpy.linalg import norm

# --- Define the RAG Query Function with return_embedding support ---
def rag_query(query, model, index, llm, return_embedding=False):
    # 1. Encode the query
    query_embedding = model.encode(query)

    # 2. Search Pinecone
    result = index.query(vector=query_embedding.tolist(), top_k=1, include_metadata=True, include_values=True)

    if not result.matches:
        doc_text = ""
        doc_embedding = None
    else:
        match = result.matches[0]
        doc_text = match.metadata.get("text", "")
        doc_embedding = match.values if hasattr(match, "values") else None

    # 3. Prepare input for LLM
    prompt = f"Context: {doc_text}\nQuestion: {query}"
    output = llm(prompt, max_length=200)
    answer = output[0]["generated_text"]

    # 4. Return depending on flag
    if return_embedding:
        return answer, doc_text, doc_embedding
    else:
        return answer

# --- Initialize models and Pinecone index ---
model = SentenceTransformer("all-MiniLM-L6-v2")
llm = pipeline("text2text-generation", model="google/flan-t5-base")
pc = Pinecone(api_key="pcsk_33qtrC_5gTrvztaWPk6Kzz4m6ZDa4vvJPeGuhWS5wRtPDow6MvMoCo7pHwsKxAHrzcFsay")
index = pc.Index("tech-support-chatbot-customer-support")

# --- Define test cases ---
test_cases = [
    {"query": "How do I reset my password?", "expected_keywords": ["reset", "password", "instructions"]},
    {"query": "My device won't turn on. What should I do?", "expected_keywords": ["device", "won't turn on", "troubleshooting"]},
    {"query": "What is the warranty period for your products?", "expected_keywords": ["warranty", "period", "products"]},
    {"query": "Can I update the firmware manually?", "expected_keywords": ["update", "firmware", "manual", "instructions"]},
    {"query": "How to connect my device to Wi-Fi?", "expected_keywords": ["connect", "device", "Wi-Fi", "steps"]},
    {"query": "What is your return policy?", "expected_keywords": ["return", "policy", "refund"]},
    {"query": "I forgot my account username, how do I retrieve it?", "expected_keywords": ["forgot", "username", "retrieve", "account"]},
    {"query": "Is this product compatible with Windows 11?", "expected_keywords": ["compatible", "Windows 11", "product"]},
    {"query": "What are the working hours for customer support?", "expected_keywords": ["working hours", "customer support", "contact"]},
    {"query": "Tell me a joke", "expected_keywords": []},
    {"query": "Who won the World Cup in 2022?", "expected_keywords": []},
    {"query": "How to resset my pasword?", "expected_keywords": ["reset", "password"]},  # typo
    {"query": "Steps for password reset", "expected_keywords": ["reset", "password"]}
]

# --- Utility Functions ---
def keyword_relevance_score(answer, keywords):
    answer_lower = answer.lower()
    found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return found / len(keywords) if keywords else None

def cosine_similarity(a, b):
    if a is None or b is None:
        return None
    return dot(a, b) / (norm(a) * norm(b))

def length_score(answer):
    return len(answer.split()) if answer else 0

# --- Evaluation Loop ---
results = []

for idx, case in enumerate(test_cases):
    query = case["query"]
    expected_keywords = case.get("expected_keywords", [])

    print(f"Test {idx+1}: Query = {query}")

    query_embedding = model.encode(query)

    start_time = time.time()
    answer, doc_text, doc_embedding = rag_query(query, model=model, index=index, llm=llm, return_embedding=True)
    duration = time.time() - start_time

    relevance = keyword_relevance_score(answer, expected_keywords)
    sim_cosine = cosine_similarity(query_embedding, doc_embedding)
    resp_len = length_score(answer)

    print(f"Answer: {answer}")
    print(f"Response Time: {duration:.3f} seconds")
    print(f"Relevance Score: {relevance}")
    print(f"Cosine Similarity: {sim_cosine}")
    print(f"Response Length (words): {resp_len}")
    print("-" * 60)

    results.append({
        "query": query,
        "answer": answer,
        "response_time_sec": duration,
        "relevance_score": relevance,
        "cosine_similarity": sim_cosine,
        "response_length_words": resp_len
    })

# --- Save CSV report ---
csv_file = "rag_chatbot_test_report.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["query", "answer", "response_time_sec", "relevance_score", "cosine_similarity", "response_length_words"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# --- Save JSON report ---
json_file = "rag_chatbot_test_report.json"
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"Test report saved to {csv_file} and {json_file}")
