# Customer Support Chatbot with Retrieval-Augmented Generation (RAG)

This project implements a **Customer Support Chatbot** using **Retrieval-Augmented Generation (RAG)** to provide intelligent and context-aware responses. The chatbot leverages a combination of **Sentence Transformers** for text embeddings, **Pinecone** for fast similarity search, and **Google's FLAN-T5** model for response generation. Additionally, it includes user authentication, chat history management, and a feedback system.

## Features

- **User Authentication:** Users can create accounts, log in, and access their chat history.
- **Context-Aware Responses:** The chatbot uses a RAG-based architecture to retrieve relevant documents and generate responses using a pre-trained model.
- **Feedback System:** Users can rate the chatbot's responses in terms of relevance and accuracy and leave additional comments.
- **Chat History:** All interactions with the chatbot are saved in a history, allowing users to access and review past conversations.
- **Real-Time Spelling Correction:** User input is corrected using **TextBlob** before being processed by the chatbot.
- **Pinecone Integration:** Fast and efficient similarity search is powered by **Pinecone**, enabling the chatbot to retrieve contextually relevant information.

## Technologies Used

- **Streamlit:** For building the web application UI.
- **Sentence Transformers:** For converting user queries and documents into embeddings.
- **Pinecone:** For vector database and similarity search.
- **Transformers:** For natural language processing and text generation with the FLAN-T5 model.
- **TextBlob:** For correcting spelling in user queries.
- **JSON:** For saving and loading chat logs and feedback data.

## Setup Instructions

### Prerequisites

- Python 3.x
- Streamlit
- Sentence Transformers
- Pinecone
- Hugging Face Transformers
- TextBlob

### Install Dependencies

1. Clone this repository or download the files.
2. Navigate to the project folder.
3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
