from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
# model=init_chat_model("mistral-medium")
model=init_chat_model("mistral-medium", temperature=0.9, max_tokens=50)
response=model.invoke("who was the first doctor in the India?")
print(response.content)