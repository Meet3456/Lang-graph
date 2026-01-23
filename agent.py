import os

from dotenv import load_dotenv
from langchain.messages import RemoveMessage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from prompts import finance_guide_prompt
from utils import stock_data_extractor

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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


class chatState(MessagesState):
    summary: str
    intent: str


def detect_intent_node(state: chatState) -> chatState:
    """
    Simple keyword-based intent detection.
    """
    # get the lastest user message content
    last_user_msg = state["messages"][-1].content.lower()

    finance_keywords = [
        "stock",
        "share",
        "price",
        "buy",
        "sell",
        "market",
        "investment",
        "portfolio",
        "nifty",
        "sensex",
    ]

    # Determine the intent bases of basic key-word matching in the users messgage
    intent = (
        "finance" if any(k in last_user_msg for k in finance_keywords) else "genral"
    )

    print(f"[INTENT DETECTED] → {intent}")

    return {"intent": intent}


def chat_node(state: chatState) -> chatState:
    """
    Uses intent to select the correct system message
    and includes conversation summary if available.
    """
    messages = []

    # inject the summary if present in the state
    if state.get("summary"):
        messages.append(
            SystemMessage(
                content=f"summary of ongoing conversation : \n {state['summary']}"
            )
        )

    # Based on the user intent, set the system prompt message of the Chatbot system and the model(with or without tools):
    if state["intent"] == "finance":
        messages.append(finance_guide_prompt.Finance_system_message)
        model = llm_with_tools
        print("[CHAT] Finance mode → tools enabled")
    else:
        messages.append(finance_guide_prompt.General_chatbot_system_message)
        model = llm
        print("[CHAT] General mode → no tools")

    # Add conversation messages (current user message):
    messages.extend(state["messages"])

    print("[CHAT NODE] Sending messages to LLM...")

    # Get the response of the current user ques (summary + SystemPrompt + Current Ques):
    response = model.invoke(messages)

    # store the response in messages:
    return {"messages": [response]}


def summarize_node(state: chatState):
    # Fetch the existing summary of conversation from the state
    existing_summary = state.get("summary", "")

    # Build summarization prompt bases on the existing_summary value

    # if existing_summary exists:
    if existing_summary:
        summary_message = HumanMessage(
            content=(
                f"""
                    Expand the summary below by incorporating the above conversation while preserving context, key points, and 
                    user intent. Rework the summary if needed. Ensure that no critical information is lost and that the 
                    conversation can continue naturally without gaps. Keep the summary concise yet informative, removing 
                    unnecessary repetition while maintaining clarity.
                    
                    Only return the updated summary. Do not add explanations, section headers, or extra commentary.

                    Existing summary:

                    {existing_summary}
                """
            )
        )

    else:
        # if no summary exists create a one for the given conversation
        summary_message = HumanMessage(
            content="""
                Summarize the above conversation while preserving full context, key points, and user intent. Your response should be concise yet detailed enough to ensure seamless continuation of the discussion. Avoid redundancy, Maintain clarity, and retain all necessary details for future exchanges.
                Only return the summarized content. Do not add explanations, section headers, or extra commentary.
            """
        )

    # Add prompt with the recent state messages:
    messages_with_summary = state["messages"] + [summary_message]

    # Invoke the llm with the summary and recent messgaes to create new summary or update the existing one:
    response = llm_with_tools.invoke(messages_with_summary)

    # Keep only the lastest(newest) 2 messages in the state["message"] rest all goes to the delete_message(as we dont need them -> summary has been created for the same)
    messages_to_delete = state["messages"][:-2]

    print("[SUMMARIZATION] Conversation summarized.")

    return {
        "summary": response.content,
        "messages": [RemoveMessage(id=m.id) for m in messages_to_delete],
    }


def route_after_chat(state: chatState):
    """
    Decide whether to:
    - run tools
    - summarize
    - end
    """
    last_msg = state["messages"][-1]

    # If LLM requested tools → go to ToolNode
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    # Otherwise check summarization
    if len(state["messages"]) > 6:
        return "summarize"

    return END


tool_node = ToolNode(toolbox)


builder = StateGraph(chatState)


# Detects whether the user's query is finance-related or general
builder.add_node("detect_intent_node", detect_intent_node)
"""
Main reasoning node:
- Injects summary (if present)
- Injects intent-based system prompt
- Decides whether to call tools or produce final output
"""
builder.add_node("chat_node", chat_node)
# Executes tool calls requested by the LLM
builder.add_node("tools", tool_node)
# Compresses conversation history when it grows too large
builder.add_node("summarize", summarize_node)


builder.add_edge(START, "detect_intent_node")
# with classified intent move to chat node where ( summary(if Present) + SystemPrompt_Intent_Based + Current Ques) is initialized
builder.add_edge("detect_intent_node", "chat_node")
"""
Based on the LLM's output, we decide:
1. Run tools (if tool_calls are present)
2. Summarize memory (if message count exceeds threshold)
3. End execution (final answer is ready)
"""
builder.add_conditional_edges(
    "chat_node",
    route_after_chat,
    {
        "tools": "tools",
        "summarize": "summarize",
        END: END,
    },
)
# After tools → go back to chat - # This allows the LLM to: Read tool results , Continue reasoning , Possibly call more tools or produce final answ
builder.add_edge("tools", "chat_node")
builder.add_edge("summarize", END)

# The checkpointer saves the graph snap shpt at various different endpoints(at each node basically)
graph = builder.compile()
