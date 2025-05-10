import streamlit as st
import os
import json
import glob
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pinecone import Pinecone
from rag_query import rag_query
from textblob import TextBlob

# ------------------------------- #
# 💾 Utility Functions
# ------------------------------- #
def get_chat_folder(username):
    folder = f"chat_logs/{username}"
    os.makedirs(folder, exist_ok=True)
    return folder

def save_chat(username, chat_id, messages):
    path = os.path.join(get_chat_folder(username), f"{chat_id}.json")
    with open(path, "w") as f:
        json.dump(messages, f)

def load_chat(username, chat_id):
    path = os.path.join(get_chat_folder(username), f"{chat_id}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def list_user_chats(username):
    folder = get_chat_folder(username)
    return sorted(glob.glob(f"{folder}/*.json"), reverse=True)

def user_exists(username):
    return os.path.exists(f"users/{username}.json")

def save_user(username, password):
    os.makedirs("users", exist_ok=True)
    with open(f"users/{username}.json", "w") as f:
        json.dump({"password": password}, f)

def validate_user(username, password):
    path = f"users/{username}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)["password"] == password
    return False

def save_feedback(username, chat_id, feedback):
    feedback_folder = f"chat_feedbacks/{username}"
    
    # Ensure the folder is created
    try:
        os.makedirs(feedback_folder, exist_ok=True)
        st.write(f"Feedback folder created at: {feedback_folder}")  # Debugging line
    except Exception as e:
        st.error(f"Error creating folder: {str(e)}")  # Error in case folder creation fails
        return
    
    feedback_path = os.path.join(feedback_folder, f"{chat_id}_feedback.json")
    
    # Save feedback data to the feedback folder
    try:
        with open(feedback_path, "w") as f:
            json.dump(feedback, f)
        st.success(f"Feedback saved at {feedback_path}")
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")

# ------------------------------- #
# ⚙️ Page Configuration
# ------------------------------- #
st.set_page_config(page_title="Customer Support Chatbot 💬", layout="wide")

# ------------------------------- #
# 🧠 Load Models and Pinecone
# ------------------------------- #
@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    llm = pipeline("text2text-generation", model="google/flan-t5-base")
    return model, llm

@st.cache_resource
def load_pinecone_index():
    pc = Pinecone(api_key="pcsk_33qtrC_5gTrvztaWPk6Kzz4m6ZDa4vvJPeGuhWS5wRtPDow6MvMoCo7pHwsKxAHrzcFsay")
    return pc.Index("chatbot-customer-support1")

model, llm = load_models()
index = load_pinecone_index()

# ------------------------------- #
# 🔐 Login / Signup
# ------------------------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Login or Signup")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if validate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.chat_id = None
                st.session_state.messages = []
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab2:
        new_user = st.text_input("Choose Username", key="new_user")
        new_pass = st.text_input("Choose Password", type="password", key="new_pass")
        if st.button("Create Account"):
            if user_exists(new_user):
                st.warning("Username already exists.")
            else:
                save_user(new_user, new_pass)
                os.makedirs(f"chat_logs/{new_user}", exist_ok=True)
                st.success("Account created! Please log in.")

    st.stop()

# ------------------------------- #
# 📜 Sidebar: List Previous Chats
# ------------------------------- #
# ------------------------------- #
# 📜 Sidebar: List Previous Chats & Feedback
# ------------------------------- #
with st.sidebar:
    st.markdown("### 📜 Your Chats")
    chat_files = list_user_chats(st.session_state.username)

    # Display previous chats as a clickable list
    for chat_file in chat_files:
        chat_name = os.path.basename(chat_file).replace(".json", "")
        if st.button(f"Chat {chat_name}", key=f"chat_{chat_name}") :
            st.session_state.chat_id = chat_name
            st.session_state.messages = load_chat(st.session_state.username, chat_name)
            st.session_state.previous_chat = True
            st.rerun()  # Rerun the app to load the chat

    # Start new chat button
    if st.button("➕ Start new chat"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.chat_id = f"chat_{timestamp}"
        st.session_state.messages = []
        st.session_state.previous_chat = False
        st.rerun()  # Rerun the app to initialize new chat

    if st.session_state.get("chat_id"):
        if st.button("🧹 Clear This Chat"):
            st.session_state.messages = []
            save_chat(st.session_state.username, st.session_state.chat_id, [])
            st.rerun()

    if st.button("🚪 Sign Out"):
        st.session_state.clear()
        st.rerun()

    # Feedback section at all times in sidebar
    if st.session_state.get("chat_id"):
        st.markdown("### 🔄 Provide Feedback for Last Response")
        relevance = st.slider("Relevance (1-5)", 1, 5, 3)
        accuracy = st.slider("Accuracy (1-5)", 1, 5, 3)
        additional_feedback = st.text_area("Additional Comments (Optional)")

        if st.button("Submit Feedback"):
            # Ensure feedback is saved after submission
            feedback = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "user_query": st.session_state.messages[-2]["content"],  # Last user query
                "assistant_response": st.session_state.messages[-1]["content"],  # Last assistant response
                "relevance": relevance,
                "accuracy": accuracy,
                "additional_feedback": additional_feedback,
            }
            save_feedback(st.session_state.username, st.session_state.chat_id, feedback)
            st.success("Thank you for your feedback!")

# ------------------------------- #
# 💬 Chat Interface (Main Page)
# ------------------------------- #
if st.session_state.get("chat_id"):
    st.title("💬 Customer Support Chatbot")
    st.caption(f"Welcome, **{st.session_state.username}**! Ask your questions below:")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User's input
    user_query = st.chat_input("Type your question here...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Correct spelling
        corrected_query = str(TextBlob(user_query).correct())

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = rag_query(corrected_query, model=model, index=index, llm=llm)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_chat(st.session_state.username, st.session_state.chat_id, st.session_state.messages)
