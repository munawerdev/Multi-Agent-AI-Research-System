from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.3)
message = HumanMessage(
    content="Say 'Groq API Key is working successfully!' if you can read this."
)
response = llm.invoke([message])

print("\n--- Test Result ---")
print(response.content)
print("-------------------\n")
