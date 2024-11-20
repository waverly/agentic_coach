# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    ToolMessage,
    SystemMessage,
    AIMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph, START, END
from typing import List, Literal
from pydantic import BaseModel


from .tools import (
    get_calendar_summary,
    save_focus_items,
    suggest_actions,
)

from .state import State
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# ... [imports and other code above]

# System prompt
template = """You are an assistant that helps users plan their week by utilizing available tools.

- Use the provided tools when necessary to fetch or save information.
- Do not guess information; if unsure, ask clarifying questions.
- Continue the conversation until the user indicates they are finished.

Available tools:
1. `get_calendar_summary`: Provides a summary of the user's calendar.
2. `save_focus_items`: Saves the user's focus items for the week.
3. `suggest_actions`: Suggests actions based on a focus item.

Remember to include any new information you learn from tool outputs in your responses."""


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


# Bind tools to the LLM
llm = ChatOpenAI(temperature=0.3)
llm_with_tool = llm.bind_tools(
    [get_calendar_summary, save_focus_items, suggest_actions]
)


def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}


def get_next_step(state):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool"
        return END  # Stop and wait for user input when no tool calls
    elif isinstance(last_message, HumanMessage):
        return "info"
    elif isinstance(last_message, ToolMessage):
        return "info"
    return END


def tool_chain(state):
    last_message = state["messages"][-1]

    # Check if there are any tool calls
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": [AIMessage(content="Let me help you with that.")]}

    # Handle all tool calls
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_messages.append(
            ToolMessage(content="Tool was called!", tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages}


memory = MemorySaver()
# Update graph
graph_builder = StateGraph(State)
graph_builder.add_node("info", info_chain)
graph_builder.add_node("tool", tool_chain)

# Add edges with conditional routing
graph_builder.set_entry_point("info")
graph_builder.add_conditional_edges("info", get_next_step)
graph_builder.add_conditional_edges("tool", get_next_step)

graph = graph_builder.compile(checkpointer=memory)
