from dateutil.parser import parse as parse_datetime
from typing import Literal, List
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
import json
from pathlib import Path

from langgraph.prebuilt import ToolNode


@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"
    else:
        raise AssertionError("Unknown city")


@tool
def get_gcal_events():
    """Use this to get Google Calendar events."""
    json_path = Path(__file__).parent.parent / "mocks" / "gcal.json"
    with open(json_path) as f:
        return json.load(f)


@tool
def get_calendar_summary() -> str:
    """Analyzes calendar events and returns a summary of the week ahead."""
    events = get_gcal_events.invoke({})
    summary = "Here's your week ahead:\n"
    for event in events["events"]:
        start_datetime = event["start"]["dateTime"]
        end_datetime = event["end"]["dateTime"]
        # Optionally, parse the datetime strings
        start_dt = parse_datetime(start_datetime)
        end_dt = parse_datetime(end_datetime)
        # Format date and time as needed
        date_str = start_dt.strftime("%B %d, %Y")
        time_str = start_dt.strftime("%I:%M %p") + " - " + end_dt.strftime("%I:%M %p")
        summary += f"- {date_str} at {time_str}: {event['summary']}\n"
    return summary


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
