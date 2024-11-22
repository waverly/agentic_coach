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
    get_competency_matrix_for_level,
    get_day_of_week,
    get_day_of_week_tool,
    get_user_context,
    get_user_context_string,
    get_user_first_name,
    get_user_first_name_tool,
    save_focus_items,
    suggest_actions,
    get_github_pull_requests,
)
from .llm import llm

from .state import State



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompt
template = """You are an assistant that helps users plan their week by utilizing available tools.

When users ask about their information:
- Use get_user_context_string for questions about level, manager, or general employee info
- Use get_user_first_name for just the user's name
- Use get_competency_matrix_for_level for role competency information

Available tools:
1. `get_calendar_summary`: Provides a summary of the user's calendar.
2. `save_focus_items`: Saves the user's focus items for the week.
3. `suggest_actions`: Suggests actions based on a focus item.
4. `get_day_of_week`: Returns the current day of the week.
5. `get_user_first_name`: Retrieves the user's first name.
6. `get_competency_matrix_for_level`: Retrieves the user's competency matrix for a given level.
7. `get_user_context_string`: Retrieves the user's employee data such as name, manager, level.
8. `get_github_pull_requests`: Retrieves recent github pull requests (PRs) authored by the user.

Remember to:
- Use the provided tools when necessary to fetch information
- Do not guess information; always use tools to fetch accurate data
- Format responses in a conversational way using the tool results"""


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages

# Tools for the LLM (returns strings)
llm_tools = [
    get_day_of_week,          
    get_user_first_name,      
    get_competency_matrix_for_level,
    get_user_context_string,
    get_github_pull_requests,
]
llm_with_tools = llm.bind_tools(llm_tools)

# New system prompt
prompt_system = """Based on the following user input, offer relevant information and continue the conversation:
{reqs}"""

def get_chatbot_messages(messages: list):
    logger.info("top of get_chatbot_messages")
    # Always include the system prompt with tools information
    messages_to_send = [SystemMessage(content=template)]
    
    # Add all non-tool messages to the conversation
    for m in messages:
        if not isinstance(m, ToolMessage):
            messages_to_send.append(m)
    
    logger.info(f"Sending {len(messages_to_send)} messages to LLM")
    return messages_to_send

def chatbot_gen_chain(state):
    logger.info('get type of messages: %s', type(state["messages"][-1]))
    logger.info("Is this a tool message? %s", isinstance(state["messages"][-1], ToolMessage))\
        
    if isinstance(state["messages"][-1], ToolMessage):
        logger.info("Here is the raw tool result: %s", state["messages"][-1])
        logger.info("Tool message found: adding to state.")
        tool_result = state["messages"][-1].content
        logger.info("tool result: %s", type(tool_result))
        logger.info("tool result: %s", tool_result)
        # Format a proper response using the tool result
        state["messages"].append(AIMessage(content=tool_result))
        return state
    
    elif isinstance(state["messages"][-1], HumanMessage):
        messages = get_chatbot_messages(state["messages"])
        response = llm_with_tools.invoke(messages)
        state["messages"].append(response)  # Just append the response
        return state
    elif not state["messages"]:
        logger.info("No messages found; skipping chatbot.")
        return state 
    else:
        logger.info("Message wasnt any discernable type")
        return state

def conversation_starter_chain(state: State):
    if state.get("starter_done", False):
        logger.info("Conversation starter chain already done.")
        return state  # Skip if already done

    logger.info("Conversation starter chain invoked!")
    user_first_name = get_user_first_name.run({})
    weekday = get_day_of_week.run({})
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

def route_based_on_human_input(state):
    user_message = next(
        (
            msg.content
            for msg in reversed(state["messages"])
            if isinstance(msg, HumanMessage)
        ),
        "",
    )
    
    if not state.get("starter_done", False):
        logger.info("About to start the conversation with conversation starter chain.")
        return "conversation_starter_chain"

    # Create a routing prompt for the LLM
    routing_prompt = """Given the following user message, determine which node to route to.
    Available nodes:
    - "cal_sum": For calendar-related queries (e.g., schedule, meetings, events)
    - "get_competency_matrix_for_level": For questions around competencies for given role
    - "get_user_context_string": For questions around user employee data such as level, name or manager (e.g., name, manager, level)
    - "chatbot": For general queries and tool usage (e.g., user info, competencies, actions)
    
    User message: "{message}"
    
    Respond with only one word: one of the given string names from the available nodes above".
    """
    
    # Get routing decision from LLM
    response = llm.invoke(routing_prompt.format(message=user_message))
    route = response.content.strip().lower()
    
    logger.info(f"LLM routing decision: {route} for message: {user_message}")
    
    # Validate the response
    if route in ["cal_sum", "chatbot"]:
        return route
    else:
        logger.warning(f"Invalid route '{route}' returned by LLM, defaulting to chatbot")
        return "chatbot"

# def route_based_on_ai_input(state):
#     ai_message = next(
#         (
#             msg.content
#             for msg in reversed(state["messages"])
#             if isinstance(msg, AIMessage)
#         ),
#         "",
#     )
    
#     # If we just showed the calendar, route to chatbot
#     if "Here's your" in ai_message and "schedule:" in ai_message:
#         return "chatbot"

#     # Create a routing prompt for the LLM
#     routing_prompt = """Given the following ai message, determine which node to route to.
#     Available nodes:
#     - "cal_sum": For calendar-related queries (e.g., schedule, meetings, events)
#     - "get_competency_matrix_for_level": For questions around competencies for given role
#     - "get_user_context": For questions around user context (e.g., name, manager, level)
#     - "chatbot": For general queries and tool usage (e.g., user info, competencies, actions)
    
#     User message: "{message}"
    
#     Respond with only one word: either "cal_sum" or "chatbot".
#     """
    
#     # Get routing decision from LLM
#     response = llm.invoke(routing_prompt.format(message=ai_message))
#     route = response.content.strip().lower()
    
#     logger.info(f"LLM routing decision: {route} for message: {ai_message}")
    
#     return "chatbot"  # Default to chatbot after calendar summary


# Tools for the ToolNode (must be properly formatted)
# Still havent really figured out what these need to return in terms of type

tools_for_node = [
    get_user_first_name,
    get_day_of_week,
    get_competency_matrix_for_level,
    get_user_context_string,
    get_github_pull_requests,
]
tool_node = ToolNode(tools=tools_for_node)


memory = MemorySaver()
graph_builder = StateGraph(State)

# Add nodes to graph
graph_builder.add_node("conversation_starter_chain", conversation_starter_chain)
# this will be replaced by a fn that synthesizes calendar, github, jira, lattice data
graph_builder.add_node("cal_sum", calendar_summary_chain)
graph_builder.add_node("chatbot", chatbot_gen_chain)
graph_builder.add_node("tools", tool_node)



graph_builder.set_entry_point("conversation_starter_chain")
graph_builder.add_conditional_edges(
    "conversation_starter_chain",
    route_based_on_human_input,
)
# graph_builder.add_conditional_edges("cal_sum", route_based_on_ai_input)
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(checkpointer=memory)