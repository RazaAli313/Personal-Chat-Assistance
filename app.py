from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage,AIMessage,AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing import Annotated,Optional,List,TypedDict
from dotenv import load_dotenv
import os
from IPython.display import display,Image,Markdown
from groq import Groq
import random
import gradio as gr


class State(TypedDict):
    messages:Annotated[List[AnyMessage],add_messages]
    

llm=Groq(base_url="https://api.groq.com",api_key=os.getenv("GROQ_API_KEY"))

def chat(oldState:State):
    # user_prompt=oldState['messages'][-1]
    formatted_messages=[]
    role=None
    for message in oldState['messages']:
        if isinstance(message,HumanMessage):
           role="user"
        else:
           role="assistant"
        formatted_messages.append({"role":role,"content":message.content})
    formatted_messages.insert(0,{"role":"system","content":"You are a helpful assistant"})
    response=llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=formatted_messages
   
    )
  
    ai=AIMessage(content=response.choices[0].message.content)
    return {"messages":[ai]}



graph_builder=StateGraph(State)


graph_builder.add_node("chat",chat)


graph_builder.add_edge(START,"chat")
graph_builder.add_edge("chat",END)


graph=graph_builder.compile()
    
def gradio_chat(user_prompt,history):
    
    messages = []
    for human, bot in history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=bot))
  
    messages.append(HumanMessage(content=user_prompt))

    state = {"messages": messages}
  

    result=graph.invoke(state)
    reply=result["messages"][-1].content
    return reply



app=gr.ChatInterface(
    fn=gradio_chat,
    title="Chat Bot, powered by Syed Muhammad Raza Ali Zaidi",
    description="Chat with Llama 3.3"
)


app.launch()
