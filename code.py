# ==================================
# Install dependencies
# ==================================
!pip install langchain langgraph groq tavily-python gradio python-dotenv langchain-groq langchain-community

# ==================================
# Imports
# ==================================
import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool


# ==================================
# Set API Keys
# ==================================
os.environ["GROQ_API_KEY"] = "PASTE API KEY HERE"
os.environ["TAVILY_API_KEY"] = "PASTE API KEY HERE"

# ==================================
# Initialize Groq LLM
# ==================================
llm = ChatGroq(
    temperature=0.6,
    model="llama-3.3-70b-versatile"  # You can switch this to 'llama3-70b-8192'
)

# ==================================
# Load tools (Math + Search)
# ==================================
# tavily_tool = TavilySearchAPIWrapper() # Deprecated way
# tools = load_tools(["llm-math"], llm=llm) # Deprecated way
# tools.append(tavily_tool) # Deprecated way

# New way to define tools for LangGraph
tavily_tool = TavilySearchResults(max_results=5)

@tool
def llm_math(query: str):
    """Tool to solve math problems."""
    from langchain.chains import LLMMathChain
    llm_math_chain = LLMMathChain.from_llm(llm)
    return llm_math_chain.run(query)

tools = [tavily_tool, llm_math]

# ==================================
# Initialize LangGraph Agent
# ==================================
agent = create_react_agent(llm, tools)


# ==================================
# Define chatbot function
# ==================================
def groq_chatbot(query):
    try:
        # Modify agent.run to use the LangGraph agent
        response = agent.invoke({"messages": [("user", query)]})
        # LangGraph agent returns a dict, extract the output
        return response['messages'][-1].content
    except Exception as e:
        return f"Error: {str(e)}"

# ==================================
# Gradio Interface
# ==================================
ui = gr.Interface(
    fn=groq_chatbot,
    inputs="text",
    outputs="text",
    title="Groq AI Agent Chatbot (LangGraph + LangChain)",
    description="A conversational AI agent powered by Groq, LangGraph, and Tavily Search."
)

ui.launch(debug=True)
