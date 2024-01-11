from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.indexes import SQLRecordManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from pandas import DataFrame
from langchain.indexes import index
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


class DataEmbedding:
    def __init__(self, settings: dict[str, Any]):
        embedding_settings = settings["embedding"]
        embedding_type = embedding_settings["type"]
        if embedding_type == "openai":
            self._embedding = OpenAIEmbeddings(api_key=embedding_settings["api_key"])

        else:
            print(f"Embedding {embedding_type} not supported.")

        database_settings = settings["database"]
        database_type = database_settings["type"]
        if database_type == "chroma":
            self._vectorstore = Chroma(embedding_function=self._embedding,
                                       persist_directory=database_settings["persist_directory"],
                                       collection_name=database_settings["collection_name"])
            self._record_manager = SQLRecordManager(database_settings["sql"]["namespace"],
                                                    db_url=database_settings["sql"]["url"])
            self._record_manager.create_schema()
            self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=database_settings["splitter"]["chunk_size"],
                                                                                  chunk_overlap=database_settings["splitter"]["chunk_overlap"])
        else:
            print(f"Database {database_type} not supported.")

        if embedding_type == "openai":
            self._qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name=embedding_settings["chat"]["model_name"],
                                                        temperature=embedding_settings["chat"]["temperature"],
                                                        openai_api_key=embedding_settings["api_key"]),
                                        chain_type=embedding_settings["chain_type"],
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
        return self._qa.invoke(query)