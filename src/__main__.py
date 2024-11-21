import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
import json
import uuid
import logging
from typing import Literal

from langgraph.graph import START, END
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from src.config import TAVILY_API_KEY, OPENAI_API_KEY

from src.chatbot.chatbot import graph, template

# Configure logging to only show WARNING and above for all loggers
logging.basicConfig(level=logging.WARNING)

# Specifically silence verbose loggers
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Only show your application's debug logs if needed
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    conversation_history = []  # Start with empty history

    def stream_graph_updates():
        for event in graph.stream({"messages": conversation_history}, config=config):
            for value in event.values():
                messages = value.get("messages", [])
                for message in messages:
                    if isinstance(message, AIMessage):
                        print("Assistant:", message.content)
                conversation_history.extend(value["messages"])

    # Initial run to start conversation
    stream_graph_updates()

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            # Add user input to history
            conversation_history.append(HumanMessage(content=user_input))
            stream_graph_updates()
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
