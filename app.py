import streamlit as st
import torch
from model_handler import load_model_and_tokenizer, generate_response
import time

# Page configuration
st.set_page_config(
    page_title="Phi-2 Fine-tuned Assistant",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for chat interface
st.markdown("""
<style>
.user-bubble {
    background-color: #e6f7ff;
    border-radius: 15px;
    padding: 10px 15px;
    margin: 5px 0;
    max-width: 80%;
    margin-left: auto;
    margin-right: 10px;
    position: relative;
    color: #000000;  /* Ensure text is black */
}
.assistant-bubble {
    background-color: #f0f0f0;
    border-radius: 15px;
    padding: 10px 15px;
    margin: 5px 0;
    max-width: 80%;
    margin-left: 10px;
    position: relative;
    color: #000000;  /* Ensure text is black */
}
.chat-container {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 200px);
    overflow-y: auto;
    padding: 10px;
    margin-bottom: 20px;
}
.stTextInput>div>div>input {
    border-radius: 20px;
}
.stButton>button {
    border-radius: 20px;
    width: 100%;
}
/* Fix for excessive vertical space */
header {
    visibility: hidden;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
h1 {
    margin-top: 0 !important;
    margin-bottom: 1rem !important;
}
/* Ensure dark mode compatibility */
.stApp {
    background-color: #121212;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# App title - made smaller to reduce vertical space
st.markdown("<h2>Phi-2 Fine-tuned Assistant</h2>", unsafe_allow_html=True)

# Load model (only once)
if not st.session_state.model_loaded:
    with st.spinner("Loading the fine-tuned model... This may take a minute."):
        model, tokenizer = load_model_and_tokenizer()
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.model_loaded = True
        st.success("Model loaded successfully!")

# Display chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f'<div class="user-bubble">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{content}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Chat input
with st.container():
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("", key="user_input", placeholder="Type your message here...")
    with col2:
        clear_button = st.button("Clear")
    
    # Process input when user submits a message
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Create a placeholder for the assistant's response
        assistant_response_placeholder = st.empty()
        
        # Generate streaming response
        full_response = ""
        
        # Display "Assistant is typing..." message
        with assistant_response_placeholder.container():
            st.markdown('<div class="assistant-bubble">Assistant is typing...</div>', unsafe_allow_html=True)
        
        # Generate response with streaming
        for token in generate_response(
            st.session_state.model, 
            st.session_state.tokenizer, 
            user_input  # Changed to pass just the current input
        ):
            full_response += token
            # Update the response in real-time
            with assistant_response_placeholder.container():
                st.markdown(f'<div class="assistant-bubble">{full_response}</div>', unsafe_allow_html=True)
            time.sleep(0.01)  # Small delay to make streaming visible
        
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Clear the input box
        st.session_state.user_input = ""
        
        # Rerun to update the UI
        st.experimental_rerun()

# Clear chat when button is pressed
if clear_button:
    st.session_state.messages = []
    st.experimental_rerun()
