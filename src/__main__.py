from langgraph.graph import START, END
from src.config import TAVILY_API_KEY, OPENAI_API_KEY

from src.chatbot.chatbot import graph


def main():
    print("Using TAVILY API Key:", TAVILY_API_KEY)
    print("Using OpenAI API Key:", OPENAI_API_KEY)

    def stream_graph_updates(user_input: str):
        for event in graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break


if __name__ == "__main__":
    main()
