import streamlit as st
import os
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
# Fixed typo: enviorn -> environ
groq_api_key = os.environ.get('GROQ_API_KEY')

# Page configuration
st.set_page_config(
    page_title="Groq Chat Application",
    page_icon=":robot_face:",
    layout="centered",
    initial_sidebar_state="expanded",
)   

def initalize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'total_messages' not in st.session_state:
        st.session_state['total_messages'] = 0
    if 'start_time' not in st.session_state:
        st.session_state['start_time'] = None
    if 'selected_persona' not in st.session_state:
        st.session_state['selected_persona'] = 'Default'

def get_custom_prompt():
    """Get Custom Prompt template based on selected persona"""
    selected_persona_name = st.session_state.get('selected_persona', 'Default')
    
    # Fixed: Renamed internal dict to avoid overwriting the variable 'persona'
    personas_dict = {
        'Default': "You are a helpful assistant.",
        'Tech Guru': "You are a tech guru who provides detailed explanations on technology topics.",
        'Friendly Companion': "You are a friendly companion who engages in casual conversation and offers emotional support.",
        'Creative Writer': "You are a creative writer who helps generate ideas for stories, poems, and other creative writing projects."
    }
    
    system_message = personas_dict.get(selected_persona_name, "You are a helpful assistant.")
    
    # Fixed: corrected input_variables and template structure
    template = system_message + """
    Current conversation:
    {history}
    Human: {input}
    Chatbot:"""
    
    return PromptTemplate(input_variables=["history", "input"], template=template)

def main(): # Fixed semicolon to colon
    initalize_session_state()
    
    with st.sidebar:
        st.title("Settings")
        model = st.selectbox(
            'Choose a model',
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        )
        conversational_memory_length = st.slider("Conversational Memory Length:", 1, 10, 5)
        
        st.session_state['selected_persona'] = st.selectbox(
            'Choose a persona',
            ['Default', 'Tech Guru', 'Friendly Companion', 'Creative Writer']
        )
        
        if st.session_state.start_time:
            st.subheader("Chat Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Messages", len(st.session_state.chat_history))
            with col2: 
                duration = datetime.now() - st.session_state.start_time
                st.metric("Chat Duration", str(duration).split('.')[0])
        
        if st.button("Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.start_time = None
            st.rerun() # Fixed: st.return() is not a function

    # Main chat interface
    st.title("Groq Chat Application")

    # Initialize memory and chain
    memory = ConversationBufferWindowMemory(k=conversational_memory_length)
    
    # Load history into memory
    for message in st.session_state.chat_history:
        memory.save_context({"input": message['human']}, {"output": message['ai']})

    groq_chat = ChatGroq(groq_api_key=groq_api_key, model=model)
    conversation = ConversationChain(memory=memory, llm=groq_chat, prompt=get_custom_prompt())

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message['human'])
        with st.chat_message("assistant"):
            st.write(message['ai'])

    # User input section
    st.markdown("---")
    user_question = st.text_input("Ask Your Question Here:", key="user_input")

    col1, col2 = st.columns([1, 5])
    with col1:
        send_button = st.button("Send")

    if send_button and user_question:
        if st.session_state.start_time is None:
            st.session_state.start_time = datetime.now()
            
        with st.spinner("Thinking..."):
            try:
                response = conversation.predict(input=user_question)
                message = {'human': user_question, 'ai': response}
                st.session_state.chat_history.append(message)
                st.rerun() # Rerun to show the new message in the history loop above
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ by [Your Name](https://your-website.com)")

if __name__ == "__main__":
    main()