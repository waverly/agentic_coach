# from langchain_community.tools.tavily_search import TavilySearchResults
import logging
import uuid
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from src.chatbot.chatbot import graph
from src.chatbot.state import State


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Interaction loop
if __name__ == "__main__":
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state: State = {"messages": [], "starter_done": False} 

    try:
        state = graph.invoke(state, config)

        # Print the initial AI message to initiate the conversation
        if "messages" in state and state["messages"]:
            initial_message = state["messages"][-1]
            if isinstance(initial_message, AIMessage):
                logger.info("Initial message content: %s", initial_message.content)
                initial_message.pretty_print()
            else:
                logger.warning("The first message is not an AIMessage.")
        else:
            logger.error("No initial message was generated.")

        while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Assistant: Goodbye!")
                break

            # Add user input to the state
            state["messages"].append(HumanMessage(content=user_input))

            # Invoke the graph and display the assistant's response
            state = graph.invoke(state, config)
            assistant_message = state["messages"][-1]  # Get the last AI message

            if state["messages"]:
                assistant_message = state["messages"][-1]
                if isinstance(assistant_message, AIMessage):
                    assistant_message.pretty_print()
                else:
                    logger.warning("Last message is not an AIMessage.")
    except Exception as e:
        logger.exception("An error occurred during the interaction loop:")
