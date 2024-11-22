# Need this here to avoid circular import
#
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.3)