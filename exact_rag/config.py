from pydantic import BaseModel
from typing import Any
from enum import Enum


class DictUnwrapper:
    def __init__(self, d: dict):
        self._d = d

    def get(self, *args: str):
        def unwrap(d: dict, key: str):
            if d:
                return d.get(key)
            else:
                return None

        current_dict = self._d
        for arg in args:
            current_dict = unwrap(current_dict, arg)
        return current_dict


class EmbeddingType(str, Enum):
    openai = "openai"
    ollama = "ollama"


class Embeddings(BaseModel):
    type: EmbeddingType
    api_key: str | None
    model: str | None
    chat_model_name: str
    chat_temperature: float
    chain_type: str
    search_type: str
    search_k: int
    search_fetch_k: int

    @classmethod
    def from_settings(cls, settings: dict[str | Any]):
        embedding_settings = settings.get("embedding")
        if embedding_settings is None:
            print("Section [embedding] not present in settings.")
            return cls()
        unwrapper = DictUnwrapper(embedding_settings)
        return cls(
            type=unwrapper.get("type"),
            api_key=unwrapper.get("api_key"),
            model=unwrapper.get("model"),
            chat_model_name=unwrapper.get("chat", "model_name"),
            chat_temperature=unwrapper.get("chat", "temperature"),
            chain_type=unwrapper.get("chain_type"),
            search_type=unwrapper.get("search", "type"),
            search_k=unwrapper.get("search", "k"),
            search_fetch_k=unwrapper.get("search", "fetch_k"),
        )


class DatabaseType(str, Enum):
    chroma = "chroma"
    elastic = "elastic"


class Databases(BaseModel):
    type: DatabaseType
    persist_directory: str | None
    url: str | None
    distance_strategy: str | None
    collection_name: str
    sql_namespace: str
    sql_url: str
    splitter_chunk_size: int
    splitter_chunk_overlap: int

    @classmethod
    def from_settings(cls, settings: dict[str | Any]):
        database_settings = settings.get("database")
        if database_settings is None:
            print("Section [database] not present in settings.")
            return cls()
        unwrapper = DictUnwrapper(database_settings)
        return cls(
            type=unwrapper.get("type"),
            persist_directory=unwrapper.get("persist_directory"),
            url=unwrapper.get("url"),
            distance_strategy=unwrapper.get("distance_strategy"),
            collection_name=unwrapper.get("collection_name"),
            sql_namespace=unwrapper.get("sql", "namespace"),
            sql_url=unwrapper.get("sql", "url"),
            splitter_chunk_size=unwrapper.get("splitter", "chunk_size"),
            splitter_chunk_overlap=unwrapper.get("splitter", "chunk_overlap"),
        )
