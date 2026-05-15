from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.agents import create_agent
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
from tools import search_entity, find_next

# 1. 加载 .env 文件
load_dotenv()

### load model ChatHuggingFace, see other models in langchain.model

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=2048,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)
### streaming output by passing parameter in ChatHuggingFace
chat_model = ChatHuggingFace(
    llm=llm,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
    )
### initialize tools
### create_agent 
agent = create_agent(model=chat_model, tools=[search_entity, find_next])
### build message
message = [{"role": "user", "content": "Aside from the Apple Remote, what other device can control the program Apple Remote was originally designed to interact with?"}]
### invoke
result = agent.invoke({"messages": message})
print(result)
