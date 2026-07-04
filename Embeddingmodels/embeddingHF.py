from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
tasks={
    "My name is hardik sachan",
    "Virat is the best cricket in the world",
    "Dhoni is bestt captain of cricket history"
}
vector=embedding.embed_documents(tasks)
print(vector)