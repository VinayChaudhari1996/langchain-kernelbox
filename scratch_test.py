import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_kernelbox import KernelBoxTool, get_session_id_from_config
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Configuration & Keys
load_dotenv("benchmarks/.env")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def agent_node(state: State, config: RunnableConfig):
    session_id = get_session_id_from_config(config)
    tool = KernelBoxTool(session_id=session_id)
    
    # 🎯 SMART SESSION UNDERSTANDING:
    # Since KernelBox holds the heavy data in memory, the LLM chat history naturally stays tiny!
    # We instruct the agent to query its own memory instead of relying on LLM context.
    system_msg = SystemMessage(content=(
        "You are a stateful AI. Your Python environment is persistent. "
        "To remember what variables exist, execute `dir()` or `%who`. "
        "Do NOT re-download data. Query the existing variables."
    ))
    
    # We pass the full valid message thread. Because KernelBox doesn't dump huge datasets 
    # into the chat, you won't suffer from token burn!
    messages_to_send = [system_msg] + state["messages"]
    
    # Bind the specific user's tool to the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", 
        temperature=0,
        api_key=os.environ.get("GOOGLE_API_KEY")
    )
    llm_with_tools = llm.bind_tools([tool])
    
    response = llm_with_tools.invoke(messages_to_send)
    return {"messages": [response]}

# 2. The Tool Execution Node
def tool_node(state: State, config: RunnableConfig):
    last_message = state["messages"][-1]
    
    # Re-initialize the tool to execute the code in the correct sandbox
    session_id = get_session_id_from_config(config)
    tool = KernelBoxTool(session_id=session_id)
    
    results = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == tool.name:
            result = tool.invoke(tool_call["args"])
            results.append({
                "role": "tool", 
                "tool_call_id": tool_call["id"], 
                "name": tool.name, 
                "content": str(result)
            })
    return {"messages": results}

def should_continue(state: State):
    return "tools" if state["messages"][-1].tool_calls else END

# 3. Build Graph
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")
graph = builder.compile()

if __name__ == "__main__":

    print("\n============================================================")
    print("Turn 1: Install Packages and Load NVIDIA Stock Data")
    print("============================================================")

    res1 = graph.invoke(
        {
            "messages": [
                (
                    "user",
                    """
Install required Python packages: yfinance and pandas.

Then:

1. Import all required libraries.
2. Fetch NVIDIA (NVDA) historical stock market data using yfinance.
3. Load stock data from January 1, 2025 up to today's latest available market date.
4. Display:
   - Latest closing price
   - Latest trading volume
   - Highest price in latest trading session
   - Lowest price in latest trading session
   - Last 5 rows of the dataframe
5. Store the dataframe in memory as `nvda_df`.
6. Ensure dataframe index is datetime format.
"""
                )
            ]
        },
        config={"configurable": {"thread_id": "user_a"}}
    )

    for m in res1["messages"][-3:]:
        print(f"{m.type}: {m.content}")



    print("\n============================================================")
    print("Turn 2: Monthly Returns Analysis (2025-2026)")
    print("============================================================")

    res2 = graph.invoke(
        {
            "messages": [
                (
                    "user",
                    """
Using the stored `nvda_df` dataframe:

1. Calculate monthly returns using Adjusted Close prices.
2. Compute monthly percentage returns from January 2025 through April 2026.
3. Create a clean dataframe with:
   - Month
   - Monthly Return (%)
4. Display the complete monthly returns table.
5. Identify:
   - Best performing month
   - Worst performing month
6. Store the result as `monthly_returns_df`.
"""
                )
            ]
        },
        config={"configurable": {"thread_id": "user_a"}}
    )

    for m in res2["messages"][-3:]:
        print(f"{m.type}: {m.content}")



    print("\n============================================================")
    print("Turn 3: Jan-Apr 2026 Returns Summary")
    print("============================================================")

    res3 = graph.invoke(
        {
            "messages": [
                (
                    "user",
                    """
Using the stored `monthly_returns_df` dataframe:

1. Filter only these months from year 2026:
   - January
   - February
   - March
   - April

2. Display a clean summary table with:
   - Month
   - Monthly Return (%)

3. Also display:
   - Best performing month
   - Worst performing month

4. Round all percentage values to 2 decimal places.
"""
                )
            ]
        },
        config={"configurable": {"thread_id": "user_a"}}
    )

    for m in res3["messages"][-3:]:
        print(f"{m.type}: {m.content}")