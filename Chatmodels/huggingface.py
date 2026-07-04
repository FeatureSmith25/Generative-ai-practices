from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.5,
    max_new_tokens=90
)
model=ChatHuggingFace(llm=llm)
res=model.invoke("who are you?")
print(res.content)