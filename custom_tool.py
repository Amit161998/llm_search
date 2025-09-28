"""
custom_tool.py

Overview:
---------
This script sets up a LangChain agent that can answer questions and retrieve information from multiple sources, including:
- Academic papers (via Arxiv)
- Wikipedia articles
- LangSmith documentation (using vector search)

It uses Groq's Llama3-8b-8192 model as the LLM backend and OpenAI for embeddings. The script loads API keys and configuration from environment variables, builds the necessary tools and retrievers, and demonstrates the agent with a sample query. This setup is useful for building advanced research assistants or chatbots that require multi-source retrieval and reasoning.
"""

# custom_tool.py
#
# This script configures a LangChain agent with tools for academic, Wikipedia, and documentation search, using Groq and OpenAI LLMs.
# Author: [Your Name]
# Date: [Update as needed]

# Import standard and third-party libraries
import os
import streamlit as st
import openai
from dotenv import load_dotenv
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_groq import ChatGroq
from langchain import hub
from langchain.agents import create_openai_tools_agent, AgentExecutor

# Load environment variables from .env
load_dotenv()

# Set up API keys for OpenAI and Groq
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv('OPENAI_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Set up LangSmith tracking (for experiment management)
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = os.getenv('LANGSMITH_PROJECT')

# Initialize Groq LLM (Llama3-8b-8192)
llm = ChatGroq(api_key=groq_api_key, model="Llama3-8b-8192")

# Set up Wikipedia and Arxiv tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# Load and split LangSmith documentation for retrieval
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200).split_documents(documents=docs)
vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vectordb.as_retriever()
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name='langsmith-search',
    description='Search any information regarding langsmith'
)

# Aggregate all tools for the agent
tools = [arxiv, wiki, retriever_tool]

# Load prompt template and create the agent
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an executor for the agent and run a sample query
agent_excecutor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print(agent_excecutor.invoke({"input": "How cancer cells can be detected ?"}))
