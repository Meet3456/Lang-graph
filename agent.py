import os

from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prompts import finance_guide_prompt
from utils import stock_data_extractor

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY_DEFAULT")

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.7
)

# Create tool bindings with additional attributes
lookup_stock = Tool.from_function(
    func=stock_data_extractor.lookup_stock_symbol,
    name="lookup_stock_symbol",
    description="Converts a company name to its stock symbol using a financial API.",
    return_direct=False,  # Return result to be processed by LLM
)

fetch_stock = Tool.from_function(
    func=stock_data_extractor.fetch_stock_data_raw,
    name="fetch_stock_data_raw",
    description="Fetches comprehensive stock data including general info and historical market data for a given stock symbol.",
    return_direct=False,
)

toolbox = [lookup_stock, fetch_stock]

llm_with_tools = llm.bind_tools(toolbox)


# defining the chat assistant node
def assistant(state: MessagesState):
    return {
        "messages": [
            llm_with_tools.invoke(
                [finance_guide_prompt.assistant_system_message] + state["messages"]
            )
        ]
    }


# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(toolbox))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()
