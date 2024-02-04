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
    temperature = 0.2
)

def single_JD_retrive(index, JD):
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(JD)
    experience_set = set()
    for i, node in enumerate(nodes):
        experience_set.add(node.text)
        
    return experience_set

def multiple_JD_retrive(index, JD_list):
    experience_set = set()
    prompt = """You job is to find best combination of experiences that fits in the JD. 
                Try to select 3 - 4 experiences that best fit in the JD and also fit in one-page resume.
                Here is the job description: """
    for i, JD in enumerate(JD_list):
        prompt += "{}.{}".format(i,JD)

    prompt += "\nHere are the experiences selected: \n"
    
    for JD in JD_list:
        experience = single_JD_retrive(index, JD)
        experience_set = experience_set.union(experience)

    for i, exp in enumerate(experience_set):
        prompt += "{}.{}".format(i,exp)

    response = llm.complete(prompt, temperature=0.3)
    return response




        

    