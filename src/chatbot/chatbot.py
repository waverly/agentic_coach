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
    get_user_first_name,
    create_synthesis_of_week
)
from .llm import llm

from .state import State



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompt
template = """You are an assistant that helps users plan their week by utilizing available tools.

When users ask about:
- Calendar or schedule: Use get_calendar_summary
- Their level, manager, or employee info: Use get_user_context_string
- Just their name: Use get_user_first_name
- Role competencies: Use get_competency_matrix_for_level
- Weekly synthesis or overview: ALWAYS use create_synthesis_of_week
- Focus items: Use save_focus_items
- Action suggestions: Use suggest_actions
- Current day: Use get_day_of_week

Available tools:
1. `get_calendar_summary`: Provides a summary of the user's calendar
2. `save_focus_items`: Saves the user's focus items for the week
3. `suggest_actions`: Suggests actions based on a focus item
4. `get_day_of_week`: Returns the current day of the week
5. `get_user_first_name`: Retrieves the user's first name
6. `get_competency_matrix_for_level`: Retrieves the user's competency matrix
7. `get_user_context_string`: Retrieves employee data (name, manager, level)
8. `create_synthesis_of_week`: Creates an in-depth synthesis of the week ahead

Remember to:
- Use the provided tools when necessary to fetch information
- If you get a tool message, make sure to extract only the content of the message and return it.
   For example, if you get a tool message with content "content=Here's your calendar summary for this week...",
   you should return "Here's your calendar summary for this week..." as the response.
"""


def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages


# Tools for the LLM (returns strings)
llm_tools = [
    get_day_of_week,          
    get_user_first_name,      
    get_competency_matrix_for_level,
    get_user_context_string,
    get_calendar_summary,
    create_synthesis_of_week
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
    logger.info('Processing message type: %s', type(state["messages"][-1]))
    logger.info('Message content: %s', state["messages"][-1])
    last_message = state["messages"][-1]

    
    if isinstance(state["messages"][-1], HumanMessage):
        logger.info("Human message detected")
        messages = get_chatbot_messages(state["messages"])
        response = llm_with_tools.invoke(messages)
        logger.info('LLM Response type: %s', type(response))
        logger.info('LLM Response: %s', response)
        state["messages"].append(response)
        return state
        
    elif isinstance(last_message, ToolMessage):
            tool_result = last_message.content
            tool_call_id = last_message.tool_call_id
            logger.info(f'Tool result: {tool_result}')
            logger.info(f'Just seeing if this is any better: {last_message.pretty_repr()}')
            logger.info(f'Processing ToolMessage with tool_call_id: {tool_call_id}')

            if not tool_call_id:
                logger.error("ToolMessage missing tool_call_id.")
                state["messages"].append(AIMessage(content="An error occurred while processing your request."))
                return state

            # Convert ToolMessage to AIMessage
            ai_message = AIMessage(tool_result)
            state["messages"].append(ai_message)
            return state
    
    else:
        logger.info("Message wasn't a recognized type")
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

def create_synthesis_of_week_chain(state):
    logger.info("Create synthesis of week chain invoked.")
    response = create_synthesis_of_week.invoke({})
    state["messages"].append(response)
    return state


def conversation_starter_chain(state: State):
    if state.get("starter_done", False):
        logger.info("Conversation starter chain already done.")
        return state  # Skip if already done

    logger.info("Conversation starter chain invoked!")
    user_first_name = get_user_first_name.run({})
    weekday = get_day_of_week.run({})
    content = f"Hi {user_first_name}, today is {weekday}. Would you like to start by getting a simple run down of what is on your calendar for this week, or do you want a more indepth synthesis of the week ahead?"

    state["messages"].append(AIMessage(content=content))
    state["starter_done"] = True  # Indicate the starter has completed
    logger.info("Conversation starter chain completed!")
    return state


def route_based_on_human_input(state):
    if not state.get("starter_done", False):
        logger.info("About to start the conversation with conversation starter chain.")
        return "conversation_starter_chain"

    user_message = next(
        (
            msg.content
            for msg in reversed(state["messages"])
            if isinstance(msg, HumanMessage)
        ),
        "",
    )
    
    logger.info(f"User message: {user_message}")

    if not user_message:
        logger.warning("No user message found; skipping routing.")
        return END
    
    logger.info("return chatbot now")
    return "chatbot"

    # Create a routing prompt for the LLM
    # routing_prompt = f"""Given the following user message, determine which node to route to.
    # Available nodes:
    # - "cal_sum": For simple calendar-related queries
    # - "create_synthesis_of_week": For a more in-depth analysis of the week ahead

    # User message: "{user_message}"

    # Respond with only one word: either "cal_sum" or "create_synthesis_of_week".
    # """
    
    # # Get routing decision from LLM
    # response = llm.invoke(routing_prompt)
    # route = response.content.strip().lower()
    # logger.info(f"LLM routing decision: {route} for message: {user_message}")

    # # Validate the response
    # if route == "cal_sum":
    #     # Ensure an AIMessage is created before invoking the tool
    #     ai_response = AIMessage(content="Fetching calendar summary...")
    #     state["messages"].append(ai_response)
    #     return "cal_sum"
    
    # elif route == "create_synthesis_of_week":
    #     logger.info("About to append ai message!!!!!!!!!")
    #     ai_response = AIMessage(content="Creating synthesis of the week...")
    #     state["messages"].append(ai_response)
    #     return "create_synthesis_of_week"
    # else:
    #     logger.warning(f"Invalid route '{route}' returned by LLM, reprompting")
    #     return "chatbot"  # Fallback to chatbot for handling ambiguous inputs

# Tools for the ToolNode (must be properly formatted)
# Still havent really figured out what these need to return in terms of type
tools_for_node = [
    get_user_first_name,
    get_day_of_week,
    get_calendar_summary,
    get_competency_matrix_for_level,
    get_user_context_string,
    create_synthesis_of_week
]
tool_node = ToolNode(tools=tools_for_node)

synthesis_node = ToolNode(tools=[create_synthesis_of_week, get_calendar_summary])

memory = MemorySaver()
graph_builder = StateGraph(State)

# Add nodes to graph
graph_builder.add_node("conversation_starter_chain", conversation_starter_chain)
# graph_builder.add_node("cal_sum", calendar_summary_chain)
# graph_builder.add_node("create_synthesis_of_week", synthesis_node)
# graph_builder.add_node("create_synthesis_of_week", create_synthesis_of_week)
graph_builder.add_node("chatbot", chatbot_gen_chain)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("conversation_starter_chain")
graph_builder.add_conditional_edges(
    "conversation_starter_chain",
    route_based_on_human_input,
)
# graph_builder.add_edge("create_synthesis_of_week", "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_conditional_edges(
#     "cal_sum",
#     lambda state: "chatbot" if not isinstance(state["messages"][-1], ToolMessage) else "chatbot"
# )
# graph_builder.add_conditional_edges(
#     "create_synthesis_of_week",
#     lambda state: "chatbot" if not isinstance(state["messages"][-1], ToolMessage) else "chatbot"
# )
# graph_builder.add_edge("chatbot", END)

# # graph_builder.add_conditional_edges("cal_sum", route_based_on_ai_input)
# graph_builder.add_conditional_edges(
#     "chatbot",
#     tools_condition,
# )
# graph_builder.add_edge("tools", "chatbot")


graph = graph_builder.compile(checkpointer=memory)




# TODO: start implementing analysis based on the data, and synthesizing with github
# generate insights and action items based on gcal, github, and lattice data

# TODO: also figure out why the router calls all the fns every time?


# def route_based_on_human_input(state):
#     if not state.get("starter_done", False):
#         logger.info("About to start the conversation with conversation starter chain.")
#         return "conversation_starter_chain"


#     user_message = next(
#         (
#             msg.content
#             for msg in reversed(state["messages"])
#             if isinstance(msg, HumanMessage)
#         ),
#         "",
#     )
    
#     logger.info(f"User message: {user_message}")
#     logger.info(f"conditions: {not user_message} {not state.get("starter_done", False)}")
    

#     # Create a routing prompt for the LLM
#     routing_prompt = """Given the following user message, determine which node to route to.
#     Available nodes:
#     - "cal_sum": For simple calendar-related queries
#     - "create_synthesis_of_week": For a more indepth analysis of the week ahead
    
#     User message: "{message}"
    
#     Respond with only one word: one of the given string names from the available nodes above".
#     """
    
#         # Get routing decision from LLM
#     response = llm.invoke(routing_prompt.format(message=user_message))
#     route = response.content.strip().lower()
#     logger.info(f"LLM routing response: {route}")
    
#     logger.info(f"LLM routing decision: {route} for message: {user_message}")
    
#     # Validate the response
#     if route in ["cal_sum", "create_synthesis_of_week"]:
#         return route
#     else:
#         logger.warning(f"Invalid route '{route}' returned by LLM, reprompting")
#         return AIMessage(content="I'm sorry, I didn't understand. Please enter a message indicate which of the two paths you'd like to pursue (calendar summary or week synthesis).")
#         # return "chatbot"
    
# def route_based_on_human_input(state):
#     if not state.get("starter_done", False):
#         logger.info("About to start the conversation with conversation starter chain.")
#         return "conversation_starter_chain"
    
#     logger.info("Routing to chatbot for tool-based response")
#     return "chatbot"

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

