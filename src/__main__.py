import json
from typing import Optional
from langgraph.graph import START, END
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from src.chatbot.tools import get_weather
from src.config import TAVILY_API_KEY, OPENAI_API_KEY

from src.chatbot.chatbot import graph, template
import uuid


def main():
    cached_human_responses = [
        "i am really concerned about opening at least 3 prs this week!",
    ]
    cached_response_index = 0
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "messages": [SystemMessage(content=template), HumanMessage(content="hi!")]
    }

    while True:
        try:
            user = input("User (q/Q to quit): ")
        except:
            user = cached_human_responses[cached_response_index]
            cached_response_index += 1
        print(f"User (q/Q to quit): {user}")
        if user in {"q", "Q"}:
            print("AI: Byebye")
            break
        output = None
        current_state = (
            initial_state
            if user == "hi!"
            else {"messages": [HumanMessage(content=user)]}
        )

        for output in graph.stream(
            current_state,
            config=config,
            stream_mode="updates",
        ):
            last_message = next(iter(output.values()))["messages"][-1]
            last_message.pretty_print()

        # if output and "prompt" in output:
        #     print("Done!")


if __name__ == "__main__":
    main()
