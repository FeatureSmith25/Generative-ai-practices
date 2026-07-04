from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

# Test Mistral Medium 3.5 model
model = ChatMistralAI(model="mistral-medium-3.5", temperature=0.7)

# Test message
message = model.invoke("Hello! What is your name?")
print("Response:", message.content)
print("\nMistral Medium 3.5 is working! ✅")
