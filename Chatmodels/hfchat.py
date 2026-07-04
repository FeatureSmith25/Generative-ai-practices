from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.chat_models import init_chat_model
model=init_chat_model("deepseek-ai/DeepSeek-V4-Pro")
response=model.invoke("explain AI in two line")
print(response.content)