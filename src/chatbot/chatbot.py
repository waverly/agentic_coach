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
from langgraph.prebuilt import ToolNode, tools_condition


from .tools import (
    get_calendar_summary,
    get_day_of_week,
    get_user_first_name,
    save_focus_items,
    suggest_actions,
    get_user_first_name,
)

from .state import State
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

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
tools = [
    get_calendar_summary,
    save_focus_items,
    suggest_actions,
    get_day_of_week,
    get_user_first_name,
]
llm = ChatOpenAI(temperature=0.3)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def conversation_starter_chain(state: State):
    user_first_name = get_user_first_name.invoke(input={})
    weekday = get_day_of_week.invoke(input={})
    return {
        "messages": [AIMessage(content=f"Hi, {user_first_name}! Today is {weekday}")]
    }


tool_node = ToolNode(tools=tools)
conversation_starter_node = ToolNode(tools=[get_day_of_week])
# data_retriever_node = ToolNode(tools=[get_calendar_summary], return_messages=True)
# action_item_node = ToolNode(tools=[suggest_actions, save_focus_items])

memory = MemorySaver()
# Update graph
graph_builder = StateGraph(State)
graph_builder.add_node("conversation_starter", conversation_starter_chain)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
# graph_builder.add_node("data_retriever", data_retriever_node)
# graph_builder.add_node("action_item", action_item_node)

graph_builder.set_entry_point("conversation_starter")

# Unconditional transitions
graph_builder.add_edge("conversation_starter", "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tools": "tools",  # If tools are needed
        END: END,  # If no tools are needed
    },
)
graph_builder.add_edge("tools", "chatbot")

# graph_builder.add_conditional_edges("data_retriever", "info")
# graph_builder.add_conditional_edges("action_item", "info")


# graph_builder.add_edge(END, "info")

graph = graph_builder.compile(checkpointer=memory)
