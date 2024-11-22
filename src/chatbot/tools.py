from datetime import datetime, timedelta
from dateutil.parser import parse as parse_datetime
from typing import Literal, List
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from langgraph.prebuilt import ToolNode

from src.mocks.types import Employee
from .llm import llm


# Lattice Data (User Context, Goals, Feedback, Reviews)
def get_user_context() -> Employee:
    """Use this to get the user's context."""
    json_path = Path(__file__).parent.parent / "mocks" / "employee_data.json"
    with open(json_path) as f:
        return json.load(f)
    
def get_competency_matrix() -> dict:
    """Use this to get the user's competency matrix."""
    json_path = Path(__file__).parent.parent / "mocks" / "competency_matrix.json"
    with open(json_path) as f:
        return json.load(f)


@tool
def get_user_first_name() -> str:
    """Use this to get the user's first name."""
    user_context = get_user_context()
    return user_context["first_name"]


@tool
def get_user_first_name_tool() -> AIMessage:
    """Use this to get the user's first name."""
    logger.info("get_user_first_name_tool called")
    user_context = get_user_context()
    return AIMessage(content=user_context["first_name"])


# Integrations (Gcal, Github, Jira)
@tool
def get_gcal_events(dummy: dict = None) -> dict: 
    """Use this to get Google Calendar events."""
    json_path = Path(__file__).parent.parent / "mocks" / "gcal.json"
    logger.info("get_gcal_events invoked")
    with open(json_path) as f:
        return json.load(f)


# This was working - keeping just in case
# @tool
# def get_calendar_summary(timeframe: Literal["last_week", "this_week"] = "this_week") -> str:
#     """Analyzes calendar events and returns a summary based on timeframe."""
#     logger.info(f"get_calendar_summary invoked for {timeframe}")
    
#     if timeframe == "last_week":
#         return AIMessage(content="Here's a summary for last week: [Placeholder for last week data].")
    
#     # Get this week's events
#     events = get_gcal_events.run({})
#     summary = "Here's your week ahead:\n"
#     for event in events["events"]:
#         start_datetime = event["start"]["dateTime"]
#         end_datetime = event["end"]["dateTime"]
#         start_dt = parse_datetime(start_datetime)
#         end_dt = parse_datetime(end_datetime)
#         date_str = start_dt.strftime("%B %d, %Y")
#         time_str = start_dt.strftime("%I:%M %p") + " - " + end_dt.strftime("%I:%M %p")
#         summary += f"- {date_str} at {time_str}: {event['summary']}\n"

#     return AIMessage(content=summary)

@tool
def get_user_context_string() -> str:
    """Use this to get the user data in a string format."""
    user_context = get_user_context()
    return str(user_context)


@tool
def get_competency_matrix_for_level(level: Literal["L1", "L2", "L3", "L4", "L5", "L6"]) -> dict:
    """Use this to get the user's competency matrix for a given level."""
    competency_json = get_competency_matrix()
    competency_prompt = f"""Given this JSON representing the competency matrix: {competency_json},
    return the competencies for the level {level}. Return this in a string format."""
    relevant_competencies = llm.invoke(competency_prompt)
    logger.info(f"Relevant competencies: {relevant_competencies}")
    logger.info(f"Type of relevant_competencies: {type(relevant_competencies)}")
    return relevant_competencies

@tool
def get_calendar_summary(week: Literal["last_week", "this_week"]) -> AIMessage:
    """
    Analyzes calendar events and returns a summary for the specified week.

    Parameters:
        week (str): "last" for last week, "this" for this week.

    Returns:
        str: Summary of events for the specified week.
    """

    events = get_gcal_events.run({})["events"]
    today = datetime(2024, 11, 18).date()

    if week.lower() == "last_week":
        start_of_week = today - timedelta(days=today.weekday() + 7)  # Last Monday
    elif week.lower() == "this_week":
        start_of_week = today - timedelta(days=today.weekday())  # This Monday
    else:
        return AIMessage(content="Invalid week selection.") 

    end_of_week = start_of_week + timedelta(days=6)  # Sunday of the specified week@tool


    end_of_week = start_of_week + timedelta(days=6)  # Sunday of the specified week
    # Filter events within the specified week
    filtered_events = [
        event
        for event in events
        if start_of_week
        <= datetime.fromisoformat(event["start"]["dateTime"]).date()
        <= end_of_week
    ]

    if not filtered_events:
        return f"No events found for {'last week' if week == 'last' else 'this week'}."

    summary = f"Here's your {'last weeks' if week == 'last' else 'this week'} schedule:"
    for event in filtered_events:
        start_info = event.get("start", {})
        start_dt = parse_datetime(start_info["dateTime"])
        end_info = event.get("end", {})
        end_dt = parse_datetime(end_info["dateTime"])
        date_str = start_dt.strftime("%B %d, %Y")
        time_str = start_dt.strftime("%I:%M %p") + " - " + end_dt.strftime("%I:%M %p")
        summary += f"- {date_str} at {time_str}: {event['summary']}\n"
    return AIMessage(content=summary)

# General Purpose Utilities
@tool
def get_day_of_week() -> str:
    """Use this to get the day of the week for a given datetime. If no datetime is provided, returns the current day of the week.

    Example:
        >>> from datetime import datetime
        >>> get_day_of_week(datetime(2024, 1, 1))  # Returns 'Monday' for New Year's Day 2024
    """
    # if date is None:
    #     print("No date provided, using current datetime")
    date = datetime.now()

    return f"""Today is {date.strftime("%A")}."""

@tool
def get_day_of_week_tool() -> AIMessage:
    """Use this to get the day of the week for a given datetime. If no datetime is provided, returns the current day of the week."""
    date = get_day_of_week.run({})
    return AIMessage(content=f"Today is {date.strftime('%A')}")


# TODO: Implement a sqlite3 db to store these items?
# and then another tool to query the db for focus items
@tool
def save_focus_items(items: List[str]) -> str:
    """Saves the user's focus items for follow-up."""
    # Mock saving to a file/database
    return f"I've noted your focus items: {', '.join(items)}"


@tool
def suggest_actions(focus_item: str) -> List[str]:
    """Suggests concrete actions based on a focus item."""
    # Mock action suggestions
    suggestions = {
        "productivity": [
            "Block out 2 hours for deep work each morning",
            "Set up a project tracking system",
            "Schedule weekly review sessions",
        ],
        "health": [
            "Schedule gym sessions",
            "Plan healthy meals",
            "Set reminders for breaks",
        ],
        "learning": [
            "Allocate 1 hour daily for study",
            "Find relevant online courses",
            "Set up practice projects",
        ],
    }
    return suggestions.get(
        focus_item.lower(),
        [
            "Create a specific plan",
            "Set measurable goals",
            "Schedule regular check-ins",
        ],
    )
