import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Option 1: Set your API key directly in the code (for testing purposes only)
os.environ["GOOGLE_API_KEY"] = "YOUR_ACTUAL_API_KEY_HERE"

# Option 2 (Recommended): Set it in your terminal before running:
# export GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY_HERE"

def test_gemini_connection():
    try:
        # Initialize the ChatGoogleGenerativeAI model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7
        )

        # Create a simple test message
        message = HumanMessage(content="Say 'API Key is working successfully!' if you can read this.")

        # Invoke the model
        response = llm.invoke([message])

        print("\n--- Test Result ---")
        print(response.content)
        print("-------------------\n")
        
    except Exception as e:
        print(f"\n❌ Connection failed. Error: {e}\n")

if __name__ == "__main__":
    test_gemini_connection()