# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import (
    ToolMessage,
    SystemMessage,
    AIMessage,
    HumanMessage,
)
import logging
from langgraph.graph import StateGraph, START, END
from typing import List, Literal
from pydantic import BaseModel
from langgraph.prebuilt import ToolNode, tools_condition
from .llm_instance import llm

from .tools import (
    get_calendar_summary,
    get_weather,
    get_day_of_week,
    get_user_first_name,
)

from .state import State

from langgraph.checkpoint.memory import MemorySaver


# System prompt
template = """You are an assistant that helps users plan their week by utilizing available tools.

- Use the provided tools when necessary to fetch or save information.
- Do not guess information; if unsure, ask clarifying questions.
- Continue the conversation until the user indicates they are finished.

Remember to include any new information you learn from tool outputs in your responses."""


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


tools = [get_calendar_summary]

llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    messages = state["messages"]

    # Only respond to human messages
    if not messages or not isinstance(messages[-1], HumanMessage):
        return {"messages": []}

    # Add system message and generate response
    messages_with_system = get_messages_info(messages)
    response = llm_with_tools.invoke(messages_with_system)
    return {"messages": [response]}


def conversation_starter_chain(state: State):
    user_first_name = get_user_first_name.invoke(input={})
    current_weekday = (
        get_day_of_week.invoke(input={}).replace("Today is ", "").strip(".")
    )
    greeting_message = f"Hi, {user_first_name}! Today is {current_weekday}."
    prompt_message = "Would you like a summary of **last week** or **this week**?"
    return {
        "messages": [
            AIMessage(content=greeting_message),
            AIMessage(content=prompt_message),
        ],
        "next": END,
    }


tool_node = ToolNode(tools=tools)

memory = MemorySaver()
# Create two separate graphs
starter_graph = StateGraph(State)
starter_graph.add_node("conversation_starter", conversation_starter_chain)
starter_graph.set_entry_point("conversation_starter")
starter = starter_graph.compile()

# Main conversation graph
main_graph = StateGraph(State)
main_graph.add_node("chatbot", chatbot)
main_graph.add_node("tools", tool_node)
main_graph.set_entry_point("chatbot")
main_graph.add_conditional_edges(
    "chatbot",
    tools_condition,
)
main_graph.add_edge("tools", "chatbot")
graph = main_graph.compile()

# Export both graphs
__all__ = ["starter", "graph"]
