from pydantic import Field, model_validator
from exact_rag.settings import Settings
from exact_rag.settings import FromDict
from typing import Annotated
from enum import Enum
from os import environ


class EmbeddingType(str, Enum):
    openai = "openai"
    ollama = "ollama"


class Embeddings(Settings):
    type: EmbeddingType = Field(description="Type of embedding (EmbeddingType).")
    api_key: str | None = Field(description="Token for openAI service.", default=None)
    model: str | None = Field(
        description="AI model (used only for ollama embedding).", default=None
    )
    chat_model_name: Annotated[str, FromDict("chat", "model_name")] = Field(
        description="Chat model."
    )
    chat_temperature: Annotated[float, FromDict("chat", "temperature")] = Field(
        description="Temperature parameter of the chat."
    )
    chain_type: str = Field(description="Langchain chain type.", default="stuff")
    search_type: Annotated[str, FromDict("search", "type")] = Field(
        description="Type fo search in database.", default="mmr"
    )
    search_k: Annotated[int, FromDict("search", "k")] = Field(
        description="Amount of documents to return."
    )
    search_fetch_k: Annotated[int, FromDict("search", "fetch_k")] = Field(
        description="Amount of documents to pass to search algorithm."
    )

    @model_validator(mode="after")
    def check_optionals(self) -> "Embeddings":
        if self.type == EmbeddingType.ollama:
            if not self.model:
                raise ValueError("For ollama embedding type you must specify a model.")
        elif self.type == EmbeddingType.openai:
            if not self.api_key:
                self.api_key = environ.get("OPENAI_API_KEY")
        return self


class DatabaseType(str, Enum):
    chroma = "chroma"
    elastic = "elastic"


class Databases(Settings):
    type: str = Field(description="Type of database (DatabaseType).")
    persist_directory: str | None = Field(
        description="Path of persistent file (used only for chroma databaase).",
        default=None,
    )
    url: str | None = Field(
        description="URL of database (used only for Elasticsearch).", default=None
    )
    distance_strategy: str | None = Field(
        description="Distance (used only fo Elasticsearch).", default=None
    )
    collection_name: str = Field(description="Name of text collection.")
    sql_namespace: Annotated[str, FromDict("sql", "namespace")] = Field(
        description="SQL duplicates database namespace."
    )
    sql_url: Annotated[str, FromDict("sql", "url")] = Field(
        description="URL of SQL duplicates database."
    )
    splitter_chunk_size: Annotated[int, FromDict("splitter", "chunk_size")] = Field(
        description="Chunk size for content splitting.", default=1000
    )
    splitter_chunk_overlap: Annotated[
        int, FromDict("splitter", "chunk_overlap")
    ] = Field(description="Overlapping size for text splitter chunks.", default=0)

    @model_validator(mode="after")
    def check_optional(self) -> "Databases":
        if self.type == DatabaseType.chroma:
            if not self.persist_directory:
                raise ValueError(
                    "For chroma database type you must specify a persist_directory."
                )
        elif self.type == DatabaseType.elastic:
            if not self.url:
                raise ValueError(
                    "For elasticsearch database type you must specify a url."
                )
            if not self.distance_strategy:
                raise ValueError(
                    "For elasticsearch database type you must specify a distance."
                )
        return self
