"""
LangChain Conversational Search Chatbot (app.py)
-------------------------------------------------

A Streamlit web application that provides a conversational chat interface powered by Groq's Llama3-8b-8192 language model, with integrated tools for real-time web, Wikipedia, and Arxiv search. The app demonstrates how to use LangChain's agent framework with Streamlit for interactive, real-time LLM applications, including session memory and agent reasoning display.

Features:
---------
- Conversational chatbot interface with persistent session memory
- Integrates DuckDuckGo, Wikipedia, and Arxiv search tools for up-to-date information
- Utilizes Groq's Llama3-8b-8192 model for generating responses
- Displays the agent's reasoning steps and actions in the Streamlit UI
- API key management via sidebar for secure, user-provided credentials
- Experiment management and tracing via LangSmith integration

How it Works:
-------------
1. Loads API keys and configuration from environment variables and sidebar input
2. Sets up search tools (DuckDuckGo, Wikipedia, Arxiv) with concise result formatting
3. Initializes a LangChain agent with the Groq LLM and the search tools
4. Maintains chat history in Streamlit session state for context-aware conversations
5. Displays both user and assistant messages, as well as the agent's reasoning steps, in the UI
6. Handles user input, runs the agent, and streams the assistant's response in real time

Usage:
------
- Enter your Groq API key in the sidebar to enable the chatbot
- Type your question in the chat input box and receive answers with supporting search results
- The agent's reasoning and tool usage are shown interactively as the response is generated

Dependencies:
-------------
- streamlit
- python-dotenv
- langchain
- langchain_groq
- langchain_community

"""

# --- Importing Required Libraries ---
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.streamlit import StreamlitCallbackHandler

# --- Environment and API Key Setup ---
# Load environment variables from .env file (for API keys and config)
load_dotenv()

# Set up API keys for OpenAI and Groq from environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Set up LangSmith tracking (for experiment management and tracing)
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')

# --- Tool Setup ---
# Configure Arxiv and Wikipedia wrappers for concise results
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# DuckDuckGo search tool for web queries
search = DuckDuckGoSearchRun(name="Search")

# --- Streamlit UI ---
st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🔎 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for settings (API key input)
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Add a button to clear chat history


def reset_chat():
    st.session_state["messages"] = [
        {"role": "assistant",
            "content": "Hi,I'm a chatbot who can search the web. How can I help you?"}
    ]


if st.sidebar.button("Clear Chat History"):
    reset_chat()

# --- Session State for Chat Memory ---
# Initialize chat history if not present
if "messages" not in st.session_state:
    reset_chat()

# Display chat history in the UI
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# --- Main Chat Input and Agent Logic ---
# Wait for user input in chat box
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Add user message to session state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Set up the Groq LLM with streaming output
    llm = ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192", streaming=True)
    # List of tools available to the agent
    tools = [search, arxiv, wiki]

    # Initialize a zero-shot agent with the tools and LLM
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True)

    # Display assistant's response in chat, with streaming and callback for agent's thoughts
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        # Run the agent on the full chat history, streaming output
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        # Add assistant's response to session state and display
        st.session_state.messages.append({'role': 'assistant', "content": response})
        st.write(response)
