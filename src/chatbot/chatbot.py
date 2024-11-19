# src/chatbot/chatbot.py
import json
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import ToolMessage
from langgraph.graph import START, END
from typing import Literal

from .state import State, graph_builder
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition

from .state import State  # Adjust the import based on your project structure

# Instantiate the OpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo")
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def create_graph():
    # Setup graph builder
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    return graph_builder.compile()  # Return the compiled graph


# Call this function to create the graph
graph = create_graph()  # This will still be available in the module
