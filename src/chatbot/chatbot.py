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


template = """Your job is to help a user plan their week.

You should start by asking them if they have any priorities for the week.

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool. Continue the conversation by calling 
the relevant tools, presenting new information to the user as needed, and asking clarifying questions. Do not end the conversation until the user 
states that they are finished."""


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""

    objective: str
    # variables: List[str]
    # constraints: List[str]
    # requirements: List[str]


llm = ChatOpenAI(temperature=0.3)
llm_with_tool = llm.bind_tools(
    [PromptInstructions, get_calendar_summary, save_focus_items, suggest_actions]
)


def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}


# New system prompt
prompt_system = """Based on the following user input, offer relevant information and continue the conversation:

{reqs}"""


# Function to get the messages for the prompt
# Will only get messages AFTER the tool call
def get_prompt_messages(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs


def prompt_gen_chain(state):
    messages = get_prompt_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


def get_state(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"


memory = MemorySaver()
workflow = StateGraph(State)
workflow.add_node("info", info_chain)
workflow.add_node("prompt", prompt_gen_chain)


@workflow.add_node
def add_tool_message(state: State):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]

    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    tool_id = tool_call["id"]

    # TODO: it is not clear to me if these tools need to be called
    # specifically or if langgraph should be able to figure out which ones to call
    # Execute the appropriate tool using .invoke()
    if tool_name == "get_calendar_summary":
        result = get_calendar_summary.invoke({})
    elif tool_name == "save_focus_items":
        result = save_focus_items.invoke({"items": tool_args["items"]})  # Pass as dict
    elif tool_name == "suggest_actions":
        result = suggest_actions.invoke({"focus_item": tool_args["focus_item"]})

    return {
        "messages": [
            ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
            )
        ]
    }


workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")

graph = workflow.compile(checkpointer=memory)
