import os
from openai import OpenAI

from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    set_global_service_context
)
from llama_index.llms import ChatMessage, OpenAILike  
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.agent import ReActAgent
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.core.llms.types import ChatMessage, MessageRole
from load_data import store_and_load_datas
from experience_rerank import multiple_JD_retrive

llm = OpenAILike(  
    api_base="http://localhost:1234/v1",  
    timeout=600,  # secs  
    api_key="loremIpsum",  
    is_chat_model=True,  
    context_window=4048,
    temperature = 0
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

if __name__ == "__main__":
    index = store_and_load_datas("resume_test.txt", "leo_resume")
    JD = open("JD.txt", "r").readlines()
    response = multiple_JD_retrive(index, JD)
    print(response)
    