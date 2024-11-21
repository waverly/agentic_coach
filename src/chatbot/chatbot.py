# from langchain_community.tools.tavily_search import TavilySearchResults
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    ToolMessage,
    SystemMessage,
    AIMessage,
    HumanMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Literal


from .tools import (
    get_calendar_summary,
    get_day_of_week,
    get_user_first_name,
    save_focus_items,
    suggest_actions,
    get_user_first_name,
)

from .state import State



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompt
template = """You are an assistant that helps users plan their week by utilizing available tools.

- Initialize the conversation with the conversation starter node. Only output the response
 from the conversation starter node one time.
- Use the provided tools when necessary to fetch or save information.
- Do not guess information; if unsure, ask clarifying questions.
- Continue the conversation until the user indicates they are finished.

Available tools:
1. `get_calendar_summary`: Provides a summary of the user's calendar.
2. `save_focus_items`: Saves the user's focus items for the week.
3. `suggest_actions`: Suggests actions based on a focus item.
4. `get_day_of_week`: Returns the current day of the week.
5. `get_user_first_name`: Retrieves the user's first name.

Remember to include any new information you learn from tool outputs in your responses."""


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


# Bind tools to the LLM
tools = [
    get_day_of_week,
    get_user_first_name,
]
llm = ChatOpenAI(temperature=0.3)
llm_with_tools = llm.bind_tools(tools)


def chatbot(state):
    logger.info("Chatbot invoked!")
    messages = get_messages_info(state["messages"])
    
    # Check if the last message is from the user
    if isinstance(state["messages"][-1], HumanMessage):
        try:
            # Simple, synchronous invocation
            response = llm_with_tools.invoke(messages)
            state["messages"].append(response)
            logger.info("Chatbot response appended to state.")
        except Exception as e:
            logger.exception("Error while getting response from LLM: %s", e)
            state["messages"].append(AIMessage(content="I'm sorry, something went wrong while generating the response."))
        
        return state
    else:
        logger.info("No new user input; skipping chatbot.")
        return state

def conversation_starter_chain(state: State):
    if state.get("starter_done", False):
        logger.info("Conversation starter chain already done.")
        return state  # Skip if already done

    logger.info("Conversation starter chain invoked!")
    user_first_name = get_user_first_name.invoke({})
    weekday = get_day_of_week.invoke({})
    content = f"Hi {user_first_name}, today is {weekday}. Would you like to discuss last week or this week?"

    state["messages"].append(AIMessage(content=content))
    state["starter_done"] = True  # Indicate the starter has completed
    logger.info("Conversation starter chain completed!")
    return state

def calendar_summary_chain(state):
    logger.info("Calendar summary chain invoked.")
    user_message = next(
        (msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), ""
    )
    if "last week" in user_message.lower():
        # Get the AIMessage directly from the tool
        response = get_calendar_summary.invoke({"week": "last_week"})
    elif "this week" in user_message.lower():
        # Get the AIMessage directly from the tool
        response = get_calendar_summary.invoke({"week": "this_week"})
    else:
        # Create a new AIMessage for the error case
        response = AIMessage(content="I'm sorry, I didn't understand. Please specify 'last week' or 'this week'.")

    # Add the response directly to state (it's already an AIMessage)
    state["messages"].append(response)
    return state




# TODO: start implementing analysis based on the data, and synthesizing with github
# generate insights and action items based on gcal, github, and lattice data

# TODO: also figure out why the router calls all the fns every time?

def route_based_on_input(state):
    user_message = next(
        (
            msg.content
            for msg in reversed(state["messages"])
            if isinstance(msg, HumanMessage)
        ),
        "",
    )

    logger.info("Router Based On Input Invoked: user message lower: %s", user_message.lower())

    if "last week" in user_message.lower() or "this week" in user_message.lower():
        logger.info("Transitioning to calendar summary based on user input.")
        return "cal_sum"  # Transition to calendar summary node

    if not state.get("starter_done", False):
        logger.info(
            "About to start the conversation with conversation starter chain."
        )
        return "conversation_starter_chain"  # Stay on conversation starter

    logger.info("No specific input; heading to chatbot")
    return "chatbot"  # Terminate if no valid input is given


tool_node = ToolNode(tools=tools)
# data_retriever_node = ToolNode(tools=[get_calendar_summary], return_messages=True)
# action_item_node = ToolNode(tools=[suggest_actions, save_focus_items])

memory = MemorySaver()
graph_builder = StateGraph(State)

# Add nodes to graph
graph_builder.add_node("conversation_starter_chain", conversation_starter_chain)
graph_builder.add_node("cal_sum", calendar_summary_chain)
graph_builder.add_node("chatbot", chatbot)

graph_builder.set_entry_point("conversation_starter_chain")
graph_builder.add_conditional_edges(
    "conversation_starter_chain",
    route_based_on_input,
)
graph_builder.add_edge("cal_sum", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(checkpointer=memory)