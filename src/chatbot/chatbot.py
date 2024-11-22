# from langchain_community.tools.tavily_search import TavilySearchResults
import json
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
    comprehensive_github_analysis,
    get_competency_matrix_for_level,
    get_day_of_week,
    get_day_of_week_tool,
    get_user_context,
    get_user_context_string,
    get_user_first_name,
    get_user_first_name_tool,
    save_focus_items,
    get_calendar_summary,
    create_synthesis_of_week,
    rethink_schedule,
    adjust_schedule,
    grow_in_career,
    get_github_prs_cache,
    what_can_coach_do,
    zoom_in,
    zoom_out,
)
from .llm import llm

from .state import State



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# System prompt
template = """You are a coach that helps a user maximize their effectiveness in their role. Effectiveness is defined by their ability to meet their goals,
and to be efficient in their use of time. It is also important that the user feels supported and fulfilled in their role.
and that they feel like they are progressing in their career.

When users ask about their information:
- Use get_user_context_string for questions about level, manager, or general employee info
- Use get_user_first_name for just the user's name
- Use get_competency_matrix_for_level for role competency information

When the user asks about zooming in or out, use the `zoom_in` or `zoom_out` tools.

When the user asks about what the coach can do, use the `what_can_coach_do` tool.

When the user asks about saving items, say you will be happy to do so and will save them
in a todo list that can be reviewed later. Use the `save_focus_items` tool.

If the user asks about their github activity, use the `comprehensive_github_analysis` tool.

Available tools:
1. `get_calendar_summary`: Provides a summary of the user's calendar.
2. `save_focus_items`: Saves the user's focus items for the week. Use this when the user asks to save their focus items.
3. `what_can_coach_do`: When the user asks about what the coach can do, use this tool.
4. `get_day_of_week`: Returns the current day of the week.
5. `get_user_first_name`: Retrieves the user's first name.
6. `get_competency_matrix_for_level`: Retrieves the user's competency matrix for a given level.
7. `get_user_context_string`: Retrieves the user's employee data such as name, manager, level.
8. `rethink_schedule`: Helps the user adjust their schedule based on their priorities.
9. `adjust_schedule`: Adjusts the user's schedule based on their priorities.
10. `grow_in_career`: Helps the user grow in their career by suggesting actionable items.
11. `quick_access_github_analysis`: Retrieves relevant github PRs from the user and sorts by most discussion, open PRs, and PRs that took the longest to merge.
12. `zoom_in`: Helps the user zoom in on a specific focus item.
13. `zoom_out`: Helps the user zoom out and think big picture about their career growth.'
14. `comprehensive_github_analysis`: Provides a comprehensive analysis of the user's github activity.
Remember to:
- Use the provided tools when necessary to fetch and synthesize information
- Do not guess information; always use tools to fetch accurate data
- After fetching data from a tool, format the response in a clear and conversational manner for the user.
- For example, if the user asks about their level, use the get_user_context_string tool, parse the json for relevant information, and then respond with "You are currently at level L4." instead of displaying raw data.
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
    create_synthesis_of_week,
    rethink_schedule,
    adjust_schedule,
    grow_in_career,
    get_github_prs_cache,
    what_can_coach_do,
    zoom_in,
    zoom_out,
    comprehensive_github_analysis,
    save_focus_items,
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
    logger.info("Is this a tool message? %s", isinstance(state["messages"][-1], ToolMessage))
    
    messages = get_messages_info(state["messages"])

    if isinstance(state["messages"][-1], ToolMessage):
        logger.info("Processing ToolMessage.")
        tool_result = state["messages"][-1].content
        user_message = next(
        (
            msg.content
            for msg in reversed(state["messages"])
            if isinstance(msg, HumanMessage)
        ),
        "",
        )
        
        # Create a prompt to format the tool data
        formatting_prompt = f"""
        You are a thoughtful and caring career coach and mentor. 
        Given the following user message: {user_message}, and the following tool result: {tool_result}, 
        provide a clear and concise response to the user.
        """
        
        # Invoke the LLM to format the response
        formatted_response = llm_with_tools.invoke([SystemMessage(content=formatting_prompt) ] + messages)
        formatted_text = formatted_response.content.strip()
        
        # Append the formatted message as AIMessage
        state["messages"].append(AIMessage(content=formatted_text))
        
        # Set the flag indicating a tool has been processed
        state["tool_processed"] = True
        logger.info("Formatted AIMessage appended in chatbot_gen_chain.")
        return state
    
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
    content = f"Hi {user_first_name}, today is {weekday}. Would you like to start by getting a simple run down of what is on your calendar for this week, or do you want a more indepth synthesis of the week ahead?"
    state["messages"].append(AIMessage(content=content))
    state["starter_done"] = True  # Indicate the starter has completed
    logger.info("Conversation starter chain completed!")
    return state

def calendar_summary_chain(state):
    logger.info("Calendar summary chain invoked.")
    response = get_calendar_summary.invoke({"week": "this_week"})
    state["messages"].append(response)
    return state

def create_synthesis_of_week_chain(state):
    logger.info("Create synthesis of week chain invoked.")
    response = create_synthesis_of_week.run({})
    state["messages"].append(response)
    return state



# TODO: start implementing analysis based on the data, and synthesizing with github
# generate insights and action items based on gcal, github, and lattice data
# TODO: also figure out why the router calls all the fns every time?
def route_based_on_human_input(state):
    if not state.get("starter_done", False):
        logger.info("About to start the conversation with conversation starter chain.")
        return "conversation_starter_chain"

    if state.get("tool_processed", False):
            logger.info("Tool processed, skipping routing.")
            state["tool_processed"] = False
            return "chatbot"
        
    user_message = next(
    (
        msg.content
        for msg in reversed(state["messages"])
        if isinstance(msg, HumanMessage)
    ),
    "",
    )
    

    # Create a routing prompt for the LLM
    routing_prompt = """Given the following user message, determine which node to route to.
    Available nodes:
    - "cal_sum": For calendar-related queries (e.g., schedule, meetings, events)
    - "create_synthesis_of_week": For questions around competencies for given role
    - "chatbot": For general queries and tool usage (e.g., user info, competencies, actions)
    
    User message: "{message}"
    
    Respond with only one one of the given string names from the available nodes above".
    """
    
    # Get routing decision from LLM
    response = llm.invoke(routing_prompt.format(message=user_message))
    route = response.content.strip().lower()
    
    logger.info(f"LLM routing decision: {route} for message: {user_message}")
    
    # Validate the response
    if route in ["cal_sum", "create_synthesis_of_week"]:
        return route
    else:
        logger.info(f"Not one of the chosen routes...defaulting to chatbot")
        return "chatbot"


# Tools for the ToolNode (must be properly formatted)
# Still havent really figured out what these need to return in terms of type

tools_for_node = [
    get_day_of_week,          
    get_user_first_name,      
    get_competency_matrix_for_level,
    get_user_context_string,
    get_calendar_summary,
    create_synthesis_of_week,
    rethink_schedule,
    adjust_schedule,
    grow_in_career,
    get_github_prs_cache,
    what_can_coach_do,
    zoom_in,
    zoom_out,
    comprehensive_github_analysis,
    save_focus_items,
]
tool_node = ToolNode(tools=tools_for_node)


memory = MemorySaver()
graph_builder = StateGraph(State)

# Add nodes to graph
graph_builder.add_node("conversation_starter_chain", conversation_starter_chain)
graph_builder.add_node("cal_sum", calendar_summary_chain)
graph_builder.add_node("create_synthesis_of_week", create_synthesis_of_week_chain)
graph_builder.add_node("chatbot", chatbot_gen_chain)
graph_builder.add_node("tools", tool_node)



graph_builder.set_entry_point("conversation_starter_chain")
graph_builder.add_conditional_edges(
    "conversation_starter_chain",
    route_based_on_human_input,
)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")

graph_builder.add_edge("cal_sum", "chatbot")
graph_builder.add_edge("create_synthesis_of_week", "chatbot")

graph = graph_builder.compile(checkpointer=memory)