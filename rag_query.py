# rag_query.py

MAX_SEQUENCE_LENGTH = 512

def rag_query(user_input, model, index, llm, top_k=3):
    """
    Retrieve relevant context from Pinecone and use an LLM to generate an answer.
    """
    # Generate embedding for the user input
    embedding = model.encode(user_input).tolist()

    # Retrieve top-k results from the index
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)

    # Combine the responses from the results
    context = "\n".join([m["metadata"]["response"] for m in results["matches"]])

    # Truncate context if it exceeds MAX_SEQUENCE_LENGTH
    if len(context.split()) > MAX_SEQUENCE_LENGTH:
        context = " ".join(context.split()[:MAX_SEQUENCE_LENGTH])

    # Create a prompt for the LLM with the truncated context and the user input
    prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"

    # Generate response using the LLM
    llm_output = llm(prompt, max_length=200)[0]["generated_text"]

    # Display the results
    print(f"Context:\n{context}\n")  # Optionally, print the context used for generating the answer
    print(f"Answer: {llm_output}\n")  # Display the generated answer

    return llm_output  # Return the answer (optional)
