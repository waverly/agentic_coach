from datetime import datetime, timedelta
from dateutil.parser import parse as parse_datetime
from typing import Literal, List
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
import json
from pathlib import Path

from langgraph.prebuilt import ToolNode

from src.mocks.types import Employee
from .llm_instance import llm


@tool
def get_weather(city: Literal["nyc", "sf"]) -> AIMessage:
    """Use this to get weather information."""
    if city == "nyc":
        return AIMessage(content="It might be cloudy in nyc")
    elif city == "sf":
        return AIMessage(content="It's always sunny in sf")
    else:
        raise AssertionError("Unknown city")


# Lattice Data (User Context, Goals, Feedback, Reviews)
def get_user_context() -> Employee:
    """Use this to get the user's context."""
    json_path = Path(__file__).parent.parent / "mocks" / "employee_data.json"
    with open(json_path) as f:
        return json.load(f)


@tool
def get_user_first_name() -> str:
    """Use this to get the user's first name."""
    user_context = get_user_context()
    return user_context["first_name"]


# Integrations (Gcal, Github, Jira)
@tool
def get_gcal_events() -> dict:
    """Use this to get Google Calendar events."""
    json_path = Path(__file__).parent.parent / "mocks" / "gcal.json"
    with open(json_path) as f:
        return json.load(f)


@tool
def get_github_data() -> dict:
    """Use this to get GitHub data."""
    json_path = Path(__file__).parent.parent / "mocks" / "github_data.json"
    with open(json_path) as f:
        return json.load(f)


@tool
def get_calendar_summary(week: Literal["last", "this"]) -> AIMessage:
    """
    Analyzes calendar events and returns a summary for the specified week.

    Parameters:
        week (str): "last" for last week, "this" for this week.

    Returns:
        str: Summary of events for the specified week.
    """

    if week.lower() in ["last"]:
        return AIMessage(content="no problem let me just get last weeks schedule for u")
    elif week.lower() == "this":
        return AIMessage(content="THIS WEEK SCHEDULE")
    else:
        return AIMessage(content="PLEASE SAY LAST OR THIS")

    events = get_gcal_events.invoke({})["events"]
    today = datetime(2024, 11, 18).date()

    if week == "last":
        start_of_week = today - timedelta(days=today.weekday() + 7)  # Last Monday
    elif week == "this":
        start_of_week = today - timedelta(days=today.weekday())  # This Monday
    else:
        return "Invalid week selection."

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
    return summary


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
def synthesize_summary(state) -> AIMessage:
    """
    Synthesizes data from GCal, GitHub, and Jira into a short summary.

    Returns:
        str: A concise summary combining information from all sources.
    """
    week = state.get("user_choice")
    print(f"Synthesizing summary for {week}")
    # Fetch data from all sources
    calendar_summary = get_calendar_summary.invoke({"week": week})
    github_issues = get_github_data.invoke({})
    # jira_tasks = get_jira_tasks.invoke({})

    # Combine data into a prompt for the LLM
    prompt = f"""
    Please synthesize the following information into a concise summary:

    **Calendar Summary:**
    {calendar_summary}

    **GitHub Issues:**
    {json.dumps(github_issues, indent=2)}

    Summary:
    """

    summary = llm.invoke([SystemMessage(content=prompt)])

    return AIMessage(content=summary)


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
