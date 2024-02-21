from typing import Any, Callable
from pandas import DataFrame

from exact_rag.config import EmbeddingType, Embeddings, DatabaseType, Databases

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.indexes import SQLRecordManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain.indexes import index
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama


class Caller:
    def __init__(
        self,
        callable: Callable[..., Any],
        arg_swap: dict[str, Any] | None = None,
        accept_only: list[str] | None = None,
    ):
        self._callable = callable
        self._arg_swap = arg_swap
        self._accept_only = accept_only

    def __call__(self, **args):
        if self._accept_only:
            args = {
                key: value for (key, value) in args.items() if key in self._accept_only
            }

        if self._arg_swap:
            args = {
                self._arg_swap.get(arg, arg): value for (arg, value) in args.items()
            }

        return self._callable(**args)


embeddings = {
    EmbeddingType.openai: Caller(OpenAIEmbeddings, accept_only=["api_key"]),
    EmbeddingType.ollama: Caller(OllamaEmbeddings, accept_only=["model"]),
}

dbs = {
    DatabaseType.chroma: Caller(
        Chroma,
        {"embedding": "embedding_function"},
        accept_only=["embedding", "persist_directory", "collection_name"],
    ),
    DatabaseType.elastic: Caller(
        ElasticsearchStore,
        {"collection_name": "index_name", "url": "es_url"},
        accept_only=[
            "embedding",
            "url",
            "collection_name",
            "distance_strategy",
            "strategy",
        ],
    ),
}

chats = {
    EmbeddingType.openai: Caller(
        ChatOpenAI,
        {
            "api_key": "openai_api_key",
            "chat_model_name": "model_name",
            "chat_temperature": "temperature",
        },
        accept_only=["chat_model_name", "chat_temperature", "api_key"],
    ),
    EmbeddingType.ollama: Caller(
        Ollama,
        accept_only=["model"],
    ),
}


class DataEmbedding:
    def __init__(self, embedding_model: Embeddings, database_model: Databases):
        embedding_type = embedding_model.type
        self._embedding = embeddings[embedding_type](**embedding_model.model_dump())
        print("Embedding initialized.")

        database_type = database_model.type
        self._vectorstore = dbs[database_type](
            embedding=self._embedding,
            **database_model.model_dump(),
        )
        print("Vectorstore initialized.")

        self._record_manager = SQLRecordManager(
            database_model.sql_namespace,
            db_url=f"sqlite:///{database_model.sql_url}",
        )
        print("Record manager initialized.")

        self._record_manager.create_schema()
        print("    schema created.")

        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=database_model.splitter_chunk_size,
            chunk_overlap=database_model.splitter_chunk_overlap,
        )
        print("Splitter initialized.")

        self._qa = RetrievalQA.from_chain_type(
            llm=chats[embedding_type](**embedding_model.model_dump()),
            chain_type="stuff",
            retriever=self._vectorstore.as_retriever(
                search_type=embedding_model.search_type,
                search_kwargs={
                    "k": embedding_model.search_k,
                    "fetch_k": embedding_model.search_fetch_k,
                },
            ),
        )
        print("Chat initialized.")

    def load(self, text: str):
        id_key = "hash"
        content_name = "text"
        dataframe = DataFrame().from_dict([{id_key: hash(text), content_name: text}])
        loader = DataFrameLoader(dataframe, page_content_column=content_name)
        data = loader.load()
        documents = self._splitter.split_documents(data)
        index(
            documents,
            self._record_manager,
            self._vectorstore,
            cleanup="incremental",
            source_id_key=id_key,
        )

    def chat(self, query: str):
        self.load(query)
        return self._qa.invoke({"query": query})
