import streamlit as st
import os
from groq import Groq
import random
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

def main():
    st.title("Groq Chat Application")
    st.sidebar.title("Select a LLM")
    
    # Added the missing closing parenthesis here
    model = st.sidebar.selectbox(
        'choose a model',
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    )
    
    conversational_memory_length = st.sidebar.slider("Conversational Memory Length:", 1, 10, 5)
    memory = ConversationBufferWindowMemory(k=conversational_memory_length)
    
    user_question = st.text_area("Ask Your Question Here:")
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({"input": message['human']}, {"output": message['ai']})

    # These must stay indented inside main()
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model=model)
    conversation = ConversationChain(memory=memory, llm=groq_chat)

    if user_question:
        response = conversation(user_question)
        message = {'human': user_question, 'ai': response['response']}
        st.session_state.chat_history.append(message)
        st.write("chatbot:", response['response'])

if __name__ == "__main__":
    main()