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
    
# RAG mocking
def get_tech_spec_data() -> dict:
    """Use this to get the tech spec data."""
    json_path = Path(__file__).parent.parent / "mocks" / "tech_spec.json"
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
    
# this is simulating a cache so that we dont have to hit the github api every time
def get_github_prs_cache() -> List[dict]:
    """Use this to get the user's github pull requests."""
    json_path = Path(__file__).parent.parent / "mocks" / "github_prs_results.json"
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


# Integration TOOLS (Gcal, Github, Jira)
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


# ACTION (write) stubs

# TODO: Implement a sqlite3 db to store these items?
# and then another tool to query the db for focus items
@tool
def save_focus_items() -> str:
    """Saves the user's focus items for follow-up."""
    # Mock saving to a file/database
    return f"Awesome! I saved those for you. I will check back in with you tomorrow to see how you are doing on these items."

# Analysis zoom-in
@tool
def create_synthesis_of_week() -> AIMessage:
    """Use this to create a synthesis of the week by synthesizing the calendar, github, and lattice data."""

    # gcal_data = "just whatever"
    gcal_data = get_gcal_events()["events"]
    jira_data = get_jira_data()
    tech_spec_data = get_tech_spec_data()["content"]
    open_prs = get_github_analysis_raw()
    
        # - Github pull requests: {github_pull_requests}
    
    synthesis_prompt = f"""Given the following data:
    - Calendar data: {gcal_data}
    - Jira data: {jira_data}
    - Tech spec data: {tech_spec_data}
    - Github analysis: {open_prs}

    You can assume the date today is November 18, 2024, so last week would begin on November 11, 2024
    while this week would begin on November 18, 2024.
    
    First, will want to help the user situate themselves, so provide a brief recap of what they did last week. You will do this by filtering through
    the calendar data, the github pull requests and the jira data to find events and tasks that happened last week. This recap should be in one short paragraph. Based on this data,
    provide percentage estimates of how much of their time was spent on categories like "feature work", "tech debt", "code reviews", "meetings", "admin", and "pto".
    
    Second, provide key insights about what their highest priority items are in their job right now: for example, if you read the tech spec and see that the tech lead is listed as "waverly",
    and that is also the name pulled from the user context, then you can infer that the user is the tech lead and they need to focus on shipping the product.
    
    Then, in a new paragraph,  ask if the user would like to zoom in and help them think through their focus items for the week, 
    or zoom out and think big picture about their role and career growth.
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
    
    analysis_of_events = llm.invoke(f"Provide a breakdown of percentage time spent on each event: {filtered_events}")
    string_of_analysis = f"In addition, here is a breakdown of how you spent your time last week: {analysis_of_events.content.strip()}"
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
  
    return_value = f"""
    {summary}
    
    {string_of_analysis}
    """
  
    return AIMessage(content=return_value)


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
def what_can_coach_do() -> AIMessage:
    """Suggests actions the Coach can help with."""
    logger.info("what_can_coach_do invoked")
    # Mock action suggestions
    template = """You are a friendly AI coach who is here to help you through thick and thin.
    Based on this list of tools, provide a bulleted list with a summary and ask the user how they would like to proceed.
    - zoom_in: Helps the user zoom-in on a specific focus item.
    - zoom_out: Helps the user zoom out and think big picture about their career growth.
    - synthesize_week: Helps the user synthesize their week and think through their focus items for the week.
    - get_calendar_summary: Helps the user understand their schedule for the week.
    - rethink_schedule: Helps the user adjust their schedule.
    - adjust_schedule: Helps the user adjust their schedule.
    - grow_in_career: Helps the user think big picture about their career growth.
    """
    
    what_can_coach_do = llm.invoke(template)
    
    what_can_coach_do_text = what_can_coach_do.content.strip()
    logger.info(f"what_can_coach_do_text: {what_can_coach_do_text}")

    return AIMessage(content=what_can_coach_do_text)

@tool
def zoom_out() -> AIMessage:
    """Use this to zoom out and help the user think big picture about their career growth."""
    logger.info("zoom_out invoked")
    user_updates = get_user_updates()
    user_context = get_user_context()
    user_goals = get_user_goals()
    competency_matrix = get_competency_matrix()
    tech_spec_data = get_tech_spec_data()["content"]
    # staff_eng_guide = get_staff_eng_guide()
    # staff_eng_guide_summary = llm.invoke(f"Provide a summary of the staff engineer guide: {staff_eng_guide}")
    most_discussion, open_prs, prs_that_took_longest_to_merge = quick_access_github_analysis()
    
    template = f"""You have access to the following user data:
    - User updates: {user_updates}
    - User context: {user_context}
    - User goals: {user_goals}
    - Tech spec data: {tech_spec_data}
    - Competency matrix: {competency_matrix}
    - Recent github activity analysis: {most_discussion}, {open_prs}, {prs_that_took_longest_to_merge}
    
    Use this data to think big picture about how the user is working on and how they are working
    towards their goals. Extract the user's level from their context and use that to compare 
    to the staff engineer guide to see how they are doing.
    
    In a new paragraph, ask the user if they want to delve further into their career growth, or zoom in on their focus items for the week.
    """
    
    response = llm.invoke(template)
    response_text = response.content.strip()
    
    return AIMessage(content=response_text)

@tool
def zoom_in() -> AIMessage:
    """Use this to help the user zoom-in and understand what they can
    do this week to have the most impact. Makes a list of actionable items to complete over the next week and prioritize them."""


    gcal_data = get_gcal_events()["events"]
    jira_data = get_jira_data()
    tech_spec_data = get_tech_spec_data()["content"]
    user_goals = get_user_goals()
    user_context = get_user_context()
    open_prs = get_github_analysis_raw()
    
    jira_data_in_progress = [ticket for ticket in jira_data if ticket["status"] == "In Progress"]
    jira_data_to_do = [ticket for ticket in jira_data if ticket["status"] == "To Do"]
    prioritize_prompt = f"""Given the following data:
    - Calendar data: {gcal_data}
    - Jira data: {jira_data}
    - Tech spec data: {tech_spec_data}
    - User goals: {user_goals}
    - User context: {user_context}
    - Open PRs: {open_prs}
    - Jira data in progress: {jira_data_in_progress}
    - Jira data to do: {jira_data_to_do}
    
    Based on this data, please provide a list of 7 highly specific actionable items 
    that the user can complete over the next week.
    Prioritize these items based on the user's goals and the timelines of the tech spec. 
    Each actionable item should be 
    specific and take less than one day to complete. 
    Address the technical tasks that need to be accomplished as well as the project management 
    and admin tasks related to their role.
    Tie each item to specific calendar events or jira tickets when possible.
    
    Some examples of actionable items:
    - Ask the user about their open PRs and github and ask what needs to be done to get them merged.
    - Look at the Jira data in progress and to do and ask the user what they need help with to get these tickets across the finish line
    - Look at the tech spec data and see if the events on the calendar align with the needs and timelines of the tech spec.
    - Look at the Jira data to do and pick the most relevant 4 tickets to tackle this week.
    - Look at the calendar data and see that it has been a while since the user last spoke with their collaborator (caleb),
        so recommend that they schedule a time to touch base with caleb to catch up and sync on their work on Lattice Coach.
    
    In a new paragraph, directly ask the user to list the focus items that are most interesting to them 
    and offer to save them in a todo list that can be reviewed later. Use the save_focus_items tool.

    """
     
    synthesis = llm.invoke(prioritize_prompt)
    synthesis_text = synthesis.content.strip()
    # logger.info(f"synthesis: {synthesis_text}")
    # return synthesis_text
    return AIMessage(content=synthesis_text)

def quick_access_github_analysis() -> tuple:
    """Use this to get a quick access list of github pull requests that the user has reviewed."""
    github_prs = get_github_prs_cache()    
    most_discussion = sorted(github_prs, key=lambda x: x["comments"], reverse=True)[0]
    open_prs = [pr for pr in github_prs if pr["state"] == "open"]
    
    # Parse the datetime strings before comparing
    def get_time_to_merge(pr):
        if pr["closed_at"] is None or pr["created_at"] is None:
            return 0
        closed_at = parse_datetime(pr["closed_at"])
        created_at = parse_datetime(pr["created_at"])
        return (closed_at - created_at).total_seconds()
    
    prs_that_took_longest_to_merge = sorted(
        [pr for pr in github_prs if pr["closed_at"] is not None], 
        key=get_time_to_merge,
        reverse=True
    )
    
    return (most_discussion, open_prs, prs_that_took_longest_to_merge)

@tool
def comprehensive_github_analysis() -> AIMessage:
    """Use this to get a comprehensive analysis of the user's github activity and analyze it."""
    github_prs = get_github_prs_cache()   
    template = f"""Given the following github pull requests: {github_prs}, provide a
    comprehensive analysis of the user's github activity. What do most of their PRs relate to?
    How long do they take to merge their PRs? Add any other relevant insights you can find."""
    
    response = llm.invoke(template)
    response_text = response.content.strip()
    return AIMessage(content=response_text)

def get_github_analysis_raw() -> str:
    """Use this to get a list of github pull requests that the user has reviewed."""
    res = quick_access_github_analysis()
    # logger.info(f"res: {res}")
    # logger.info(f"most_discussion: {most_discussion}")
    # logger.info(f"open_prs: {open_prs}")
    # logger.info(f"prs_that_took_longest_to_merge: {prs_that_took_longest_to_merge}")    
    
    #     - Most discussion: {most_discussion}
    # - Open PRs: {open_prs}
    # - PRs that took longest to merge: {prs_that_took_longest_to_merge}
    
    template = """Analyze the following github pull requests:
    {res}
    
    And provide a health check of how the user is doing on github. Does it seem like they are contributing to the codebase?
    Are they responsive to feedback? Are they merging their PRs in a timely manner? Are there any areas where they could improve?
    Are their PRs generating discussion? Do they have a good ratio of PRs merged to PRs reviewed?

    In general, what do most of their PRs relate to? Are they mostly feature work, or mostly admin tasks?

    Return a list of 3 actionable items that the user can complete to improve their github contributions
    based on the competency matrix for L4s Software engineers above
    """
    
    analysis = llm.invoke(template)
    analysis_text = analysis.content.strip()
    return analysis_text



@tool
def get_github_analysis() -> AIMessage:
    """Use this to get a list of github pull requests that the user has reviewed."""
    res = quick_access_github_analysis()
    logger.info(f"res: {res}")
    # logger.info(f"most_discussion: {most_discussion}")
    # logger.info(f"open_prs: {open_prs}")
    # logger.info(f"prs_that_took_longest_to_merge: {prs_that_took_longest_to_merge}")    
    
    #     - Most discussion: {most_discussion}
    # - Open PRs: {open_prs}
    # - PRs that took longest to merge: {prs_that_took_longest_to_merge}
    
    template = """Analyze the following github pull requests:
    {res}
    
    And provide a health check of how the user is doing on github. Does it seem like they are contributing to the codebase?
    Are they responsive to feedback? Are they merging their PRs in a timely manner? Are there any areas where they could improve?
    Are their PRs generating discussion? Do they have a good ratio of PRs merged to PRs reviewed?

    In general, what do most of their PRs relate to? Are they mostly feature work, or mostly admin tasks?

    Return a list of 3 actionable items that the user can complete to improve their github contributions
    based on the competency matrix for L4s Software engineers above
    """
    
    analysis = llm.invoke(template)
    analysis_text = analysis.content.strip()
    return AIMessage(content=analysis_text)


# Analysis zoom-out
@tool
def grow_in_career() -> AIMessage:
    """Use this to help the user grow in their career."""
    user_updates = get_user_updates()
    user_context = get_user_context()
    user_goals = get_user_goals()
    competency_matrix = get_competency_matrix()
    staff_eng_guide = get_staff_eng_guide()
    grow_prompt = f"""You have access to the following user data:
    - User updates: {user_updates}
    - User context: {user_context}
    - User goals: {user_goals}
    - Competency matrix: {competency_matrix}
    - Staff engineer guide: {staff_eng_guide}
    
    For an L4 engineer, you can look at the staff engineer guide to see what are the main responsibilities of a Staff engineer.
    Then, use that to do an analysis of how the user is currently doing in comparison. Give specific examples of how they are doing well and how they can improve.
    and constructively challenge them with specific examples of how they can grow in their career, like:
    - It doesnt seem like you spend a lot of time on cross-functional collaboration.
    - It seems like you dont spend enough time on technical tasks, and you should probably spend more time coding.
    - It seems like you dont spend enough time on admin tasks, and you should probably spend more time doing admin tasks.
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



# Actual third  party integrations (invoked only once and cached for demo)
@tool
def get_github_pull_requests() -> List[dict]:
    """Gets recent github pull requests (PRs) for the authenticated user."""
    logger.info("get_github_pull_requests invoked")
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
        "closed_at": r.closed_at,
        "updated_at": r.updated_at,
        "state": r.state,
        "html_url": r.html_url,
        "body": r.body,
        "comments": r.comments,
        # There's a lot of text in "body" (the PR description), so omitting for now,
        # but there's probably great signal in here for the LLM to build context.
        # "body": r.body,
    } for r in results[:30]] # <-- The per_page arg isn't working, so faking a limit here.

    
    # Write results to mock data file
    json_path = Path(__file__).parent.parent / "mocks" / "github_prs_results.json"
    with open(json_path, "w") as f:
        json.dump(results_list, f, indent=2, default=str)

    github_client.close()

    return results_list

# my PR review comments