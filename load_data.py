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

service_context = ServiceContext.from_defaults(
        llm=llm,
        chunk_size=1024,
        embed_model = "local",
)

set_global_service_context(service_context)

def preprocess(resume_file_path):
    f = open(resume_file_path, "r")
    lines = f.readlines()
    sections = []
    current_section = []

    for line in lines:
        if line.strip() == '':  # Check for empty line
            sections.append(''.join(current_section))
            current_section = []
        else:
            current_section.append(line)

    # Add the last section if it's not empty
    if current_section:
        sections.append(''.join(current_section))

    documents = [Document(text=t,metadata={"file_name":i}) for i, t in enumerate(sections)]

    return documents

def store_and_load_datas(resume_file_path, storage_path):
    try:
        storage_context = StorageContext.from_defaults(persist_dir = storage_path)
        # load index
        index = load_index_from_storage(storage_context)
    except:
        documents = preprocess(resume_file_path)
        index = VectorStoreIndex.from_documents(documents, service_context = service_context)
        index.storage_context.persist(storage_path)

    return index



