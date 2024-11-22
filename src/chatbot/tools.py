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
from github import Github
from github import Auth
from src.config import GITHUB_ACCESS_TOKEN


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
    
def get_tech_spec_data() -> dict:
    """Use this to get the tech spec data."""
    json_path = Path(__file__).parent.parent / "mocks" / "tech_spec.json"
    with open(json_path) as f:
        return json.load(f)

def get_user_goals() -> dict:
    """Use this to get the user's goals."""
    json_path = Path(__file__).parent.parent / "mocks" / "user_goals.json"
    with open(json_path) as f:
        return json.load(f)
    
def get_user_updates() -> dict:
    """Use this to get the user's updates."""
    json_path = Path(__file__).parent.parent / "mocks" / "user_updates.json"
    with open(json_path) as f:
        return json.load(f)
    
def get_staff_eng_guide() -> str:
    """Use this to get the staff engineer guide."""
    json_path = Path(__file__).parent.parent / "mocks" / "staff_eng.py"
    with open(json_path) as f:
        return f.read()

# Integrations (Gcal, Github, Jira)

def get_jira_data() -> dict:
    """Use this to get the user's Jira data."""
    json_path = Path(__file__).parent.parent / "mocks" / "jira.json"
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
def get_gcal_events() -> dict: 
    """Use this to get Google Calendar events."""
    json_path = Path(__file__).parent.parent / "mocks" / "gcal.json"
    logger.info("get_gcal_events invoked")
    with open(json_path) as f:
        return json.load(f)

@tool
def get_user_context_string() -> str:
    """Use this to get the user data in a string format."""
    user_context = get_user_context()
    return str(user_context)


@tool
def get_competency_matrix_for_level() -> dict:
    """Use this to get the competency matrix related to the user's level."""
    user_context = get_user_context()
    level = user_context["level"]
    logger.info(f"level: {level}")
    """Use this to get the user's competency matrix for a given level."""
    competency_json = get_competency_matrix()
    competency_prompt = f"""Given this JSON representing the competency matrix: {competency_json},
    return the competencies for the level {level}. Return this in a string format."""
    relevant_competencies = llm.invoke(competency_prompt)
    logger.info(f"Relevant competencies: {relevant_competencies}")
    logger.info(f"Type of relevant_competencies: {type(relevant_competencies)}")
    return relevant_competencies

@tool
def create_synthesis_of_week() -> AIMessage:
    """Use this to create a synthesis of the week by synthesizing the calendar, github, and lattice data."""

    gcal_data = "just whatever"
    # gcal_data = get_gcal_events()["events"]
    jira_data = get_jira_data()
    tech_spec_data = get_tech_spec_data()["content"]

    synthesis_prompt = f"""Given the following data:
    - Calendar data: {gcal_data}
    - Jira data: {jira_data}
    - Tech spec data: {tech_spec_data}
    
    You can assume the date today is November 18, 2024, so last week would begin on November 11, 2024
    while this week would begin on November 18, 2024.
    
    First, will want to help the user situate themselves, so provide a brief recap of what they did last week. You will do this by filtering through
    the calendar data and the jira data to find events and tasks that happened last week. This recap should be in one short paragraph. Based on this data,
    provide percentage estimates of how much of their time was spent on categories like "feature work", "tech debt", "code reviews", "meetings", "admin", and "pto".
    
    Second, provide key insights about what their highest priority items are in their job right now: for example, if you read the tech spec and see that the tech lead is listed as "waverly",
    and that is also the name pulled from the user context, then you can infer that the user is the tech lead and they need to focus on shipping the product.
    
    Third, filter through the Jira data and Calendar data to create a synthesis of the week ahead. Output this 
    synthesis of the week ahead in bullet points. Make sure to cite the data sources in the synthesis. Weave in the
    timelines and deliverables of the tech spec as well.
    
    Then, in a new paragraph,  ask if the user would like 
    help generating a list of actionable items to complete over the next week and prioritizing them. 
     """
     
    synthesis = llm.invoke(synthesis_prompt)
    synthesis_text = synthesis.content.strip()
    # logger.info(f"synthesis: {synthesis_text}")
    # return synthesis_text
    return AIMessage(content=synthesis_text)

@tool
def get_calendar_summary(week: Literal["last_week", "this_week"]) -> AIMessage:
    """
    Analyzes calendar events and returns a summary for the specified week.

    Parameters:
        week (str): "last" for last week, "this" for this week.

    Returns:
        str: Summary of events for the specified week.
    """

    events = get_gcal_events()["events"]
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
def prioritize_tasks() -> AIMessage:
    """Use this to help the user make a list of actionable items to complete over the next week and prioritize them."""


    gcal_data = get_gcal_events()["events"]
    jira_data = get_jira_data()
    tech_spec_data = get_tech_spec_data()["content"]
    user_goals = get_user_goals()
    user_context = get_user_context()

    prioritize_prompt = f"""Given the following data:
    - Calendar data: {gcal_data}
    - Jira data: {jira_data}
    - Tech spec data: {tech_spec_data}
    - User goals: {user_goals}
    - User context: {user_context}
    
    Based on this data, please provide a list of 7 actionable items that the user can complete over the next week.
    Prioritize these items based on the user's goals and the timelines of the tech spec. Each actionable item should be 
    specific and take less than one day to complete. Tie each item to specific calendar events or jira tickets when possible. If
    no artifacts exist, just make sure the actionable item is specific and achievable.
    
    Then, in a new paragraph, directly ask if they need help doing any of these tasks. Specifically mention that you have the ability to
    write updates in Lattice, and can help them schedule time in gcal.
    
    """
     
    synthesis = llm.invoke(prioritize_prompt)
    synthesis_text = synthesis.content.strip()
    # logger.info(f"synthesis: {synthesis_text}")
    # return synthesis_text
    return AIMessage(content=synthesis_text)

@tool
def grow_in_career() -> AIMessage:
    """Use this to help the user grow in their career."""
    user_updates = get_user_updates()
    user_context = get_user_context()
    user_goals = get_user_goals()
    competency_matrix = get_competency_matrix()
    staff_eng_guide = get_staff_eng_guide()
    grow_prompt = f""" You have access to the following user data:
    - User updates: {user_updates}
    - User context: {user_context}
    - User goals: {user_goals}
    - Competency matrix: {competency_matrix}
    - Staff engineer guide: {staff_eng_guide}
    
    For an L4 engineer, you can look at the staff engineer guide to see what are the main responsibilities of a Staff engineer.
    Then, use that to do an analysis of how the user is currently doing in comparison.
    Analyze ways in which this user can be more effective in their role and at their level, and suggest actionable items to help them.
    For example, you could suggest that they schedule more 1:1s with their product manager, 
    or that they schedule time to run QA sessions with their team for an important milestone in the tech spec. Make sure they are working towards some of their goals, and doing admin tasks such as writing updates in Lattice.
    Provide a list of actionable items that the user can complete to grow in their career.
    """
    

    grow = llm.invoke(grow_prompt)
    grow_text = grow.content.strip()
    # logger.info(f"synthesis: {synthesis_text}")
    # return synthesis_text
    return AIMessage(content=grow_text)

@tool
def rethink_schedule() -> AIMessage:
    """Use this to help the user adjust their schedule."""
    
    gcal_data = get_gcal_events()["events"]
    schedule_prompt = f"""Listen to the user input and extract their priority.
    Then, look at their calendar data ({gcal_data}) and suggest a time that works for them to complete the task.
    Offer a range of times, and ask if any work arounds are possible.
    """
    schedule = llm.invoke(schedule_prompt)
    schedule_text = schedule.content.strip()
    return AIMessage(content=schedule_text)

@tool
def adjust_schedule(state) -> AIMessage:
    """Use this to help the user adjust their schedule."""
    gcal_data = get_gcal_events()["events"]

    adjust_prompt = f"""based on the previous conversation ({state["messages"]}), please help the user adjust their schedule.
    get their current schedule, and then rewrite the gcal json to reflect the changes as discussed.
    
    Here is the current schedule: {gcal_data}
    
    When you are done, say "Here is the updated schedule:" and then output the updated gcal json.
    """
    adjust = llm.invoke(adjust_prompt)
    adjust_text = adjust.content.strip()
    return AIMessage(content=adjust_text)

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

@tool
def get_github_pull_requests() -> List[dict]:
    """Gets recent github pull requests (PRs) for the authenticated user."""

    # In Python, the results look like:
    #
    # [
    #    {
    #         'title': 'Lattice Assistant: QA Portal uses new line delimited json (chunked streaming) instead of regex parsing',
    #         'created_at': datetime.datetime(2024, 11, 18, 20, 53, 51, tzinfo=datetime.timezone.utc),
    #         'state': 'closed',
    #         'html_url': 'https://github.com/latticehr/lattice/pull/88095'
    #    },
    #    ...
    # ]
    
    # https://docs.github.com/en/rest/search/search?apiVersion=2022-11-28

    auth = Auth.Token(GITHUB_ACCESS_TOKEN)
    github_client = Github(auth=auth)

    results = github_client.search_issues(query="repo:latticehr/lattice is:pr author:waverly")
    results_list = [{
        "title": r.title,
        "created_at": r.created_at,
        "state": r.state,
        "html_url": r.html_url,
        # There's a lot of text in "body" (the PR description), so omitting for now,
        # but there's probably great signal in here for the LLM to build context.
        # "body": r.body,
    } for r in results[:15]] # <-- The per_page arg isn't working, so faking a limit here.

    github_client.close()

    return results_list

# my PR review comments