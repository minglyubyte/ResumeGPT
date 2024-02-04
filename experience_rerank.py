import os
from openai import OpenAI

from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
    VectorStoreIndex, 
    SimpleDirectoryReader,
    Document
)
from llama_index.llms import ChatMessage, OpenAILike  
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.agent import ReActAgent
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.core.llms.types import ChatMessage, MessageRole
from llama_index.readers import StringIterableReader

llm = OpenAILike(  
    api_base="http://localhost:1234/v1",  
    timeout=600,  # secs  
    api_key="loremIpsum",  
    is_chat_model=True,  
    context_window=4048,
    temperature = 0
)

def single_JD_retrive(index, JD):
    retriever = index.as_retriever()
    nodes = retriever.retrieve(JD)
    prompt = "Here are expereinces retrived for this JD - {}:\n".format(JD)
    for i, node in enumerate(nodes):
        prompt += "{}.{}\n\n".format(i, node.text)
    prompt += """Please choose one that best fits the JD based on experience relavance and experience date. 
    Do not return any comments or explainantion but just return the experience itself."""
    response = llm.complete(prompt, temperature=0.1)
    return "\n".join(response.text.split("\n")[1:])

def multiple_JD_retrive(index, JD_list):
    prompt = "You job is to find best combination of experiences that fits in the JD. Here is the job description: \n"
    for i, JD in enumerate(JD_list):
        prompt += "{}.{}".format(i,JD)

    prompt += "\nHere are the experiences selected: \n"
    for i, JD in enumerate(JD_list):
        experience = single_JD_retrive(index, JD)
        prompt += "{}.{}".format(i,experience)

    prompt += "Choose the best combination of experiences that fit in JD and also would fit in one-page resume."
    response = llm.complete(prompt, temperature=0.1)
    return response




        

    