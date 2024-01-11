from typing import Any, Callable

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.indexes import SQLRecordManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from pandas import DataFrame
from langchain.indexes import index
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama


class Caller:
    def __init__(self, callable: Callable[..., Any], arg_swap: dict[str, Any] | None = None, accept_only: list[str] | None = None):
        self._callable = callable
        self._arg_swap = arg_swap
        self._accept_only = accept_only

    def __call__(self, **args):
        if self._accept_only:
            args = {key: value for (key, value) in args.items() if key in self._accept_only}

        if self._arg_swap:
            args = {self._arg_swap.get(arg, arg): value for (arg, value) in args.items()}

        return self._callable(**args)


embeddings = {"openai": Caller(OpenAIEmbeddings, accept_only=["api_key"]),
              "ollama": Caller(OllamaEmbeddings, accept_only=["model"])}

dbs = {"chroma": Caller(Chroma, {"embedding": "embedding_function"}, accept_only=["embedding", "persist_directory", "collection_name"]),
       "elastic": Caller(ElasticsearchStore, {"collection_name": "index_name", "url": "es_url"} ,accept_only=["embedding", "url", "collection_name", "distance_strategy", "strategy"])}

chats = {"openai": Caller(ChatOpenAI, {"api_key": "openai_api_key"}, accept_only=["model", "temperature", "api_key"]),
         "ollama": Caller(ChatOllama, accept_only=["model", "temperature"])}


class DataEmbedding:
    def __init__(self, settings: dict[str, Any]):
        embedding_settings = settings["embedding"]
        embedding_type = embedding_settings["type"]
        self._embedding = embeddings[embedding_type](**embedding_settings)

        database_settings = settings["database"]
        database_type = database_settings["type"]

        self._vectorstore = dbs[database_type](embedding=self._embedding, **database_settings, strategy=ElasticsearchStore.ExactRetrievalStrategy())

        self._record_manager = SQLRecordManager(database_settings["sql"]["namespace"],
                                                db_url=database_settings["sql"]["url"])
        self._record_manager.create_schema()
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=database_settings["splitter"]["chunk_size"],
                                                                              chunk_overlap=database_settings["splitter"]["chunk_overlap"])

        self._qa = RetrievalQA.from_chain_type(llm=chats[embedding_type](**embedding_settings["chat"], **embedding_settings),
                                               chain_type="stuff",
                                               retriever=self._vectorstore.as_retriever(search_type=embedding_settings["search"]["type"],
                                                                                        search_kwargs={'k': embedding_settings["search"]["k"],
                                                                                                       'fetch_k': embedding_settings["search"]["fetch_k"]}))

    def load(self, text: str):
        id_key = "hash"
        content_name = "text"
        dataframe = DataFrame().from_dict([{id_key: hash(text),
                                            content_name: text}])
        loader = DataFrameLoader(dataframe,
                                 page_content_column=content_name)
        data = loader.load()
        documents = self._splitter.split_documents(data)
        index(documents,
              self._record_manager,
              self._vectorstore,
              cleanup="incremental",
              source_id_key=id_key)


    def chat(self, query: str):
        return self._qa.invoke({"query":query})