from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
load_dotenv()
model=ChatMistralAI(model="mistral-small-2506",temperature=0.9)
print("Choose your AI Mode")
print("Press 1 for Angry Mode")
print("Press 2 for Funny Mode")
print("Press 3 for Sad Mode")
choice=int(input("Tell your response:- "))
if choice==1:
        mode="You are an angry AI agent, you respond aggresively and impatiently"
elif choice==2:
        mode="You are an funny AI agent, you respond with humor and jokes"
elif choice==3:
        mode="You are an sad AI agent, you respond in a depressed and emotional tone"
massages=[
    SystemMessage(content=mode)
]

print("------------- welcome type 0 to exit the application ---------------")

while True:
    prompt=input("You : ")
    massages.append(HumanMessage(content=prompt))
    if prompt=="0":
        break
    res=model.invoke(massages)
    massages.append(AIMessage(content=res.content))
    print("Bot :", res.content)

# print(massages)