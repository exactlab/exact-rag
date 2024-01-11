import toml
from typing import Any

from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.indexes import SQLRecordManager
from langchain.indexes import index
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pandas import DataFrame

from langchain_openai import OpenAIEmbeddings


class ChromaBuilder(Chroma):
    def __init__(params: dict[str, Any], *, embedding):
        super.__init__(**params, embedding_function=embedding)


class ElasticBuilder(ElasticsearchStore):
    def __init__(params: dict[str, Any], *, embedding):
        super.__init__(**params, embedding=embedding)
    


embeddings = {"openai": OpenAIEmbeddings}
databases = {"chroma": ChromaBuilder, "elastic": ElasticBuilder}


class WrongNumberOfSelections(Exception):
    def __init__(self,
                parameter: str,
                present: int,
                expected: int = 1):
        self.parameter = parameter
        self.present = present
        self.excpected = expected


def extract_unique_setting(settings: dict[str, str | None], key: str) -> dict[str, str | None]:
    key_settings = settings[key]
    key_settings_n = len(key_settings)
    if key_settings_n != 1:
        raise WrongNumberOfSelections(key, key_settings_n)
    
    return key_settings


class DataEmbedding:
    def __init__(self, settings: dict[str, str | None]):
        # parameter extraction
        try:
            embedding_settings = extract_unique_setting(settings, "embedding")
            database_settings = extract_unique_setting(settings, "database")

        except KeyError as key:
            print(f"Key {key.args[0]} not present.")

        except WrongNumberOfSelections as selection:
            print(f"Parameter {selection.parameter}: expected {selection.excpected}, present {selection.present}.")


        try:
            embedding = list(embedding_settings.keys())[0]
            embedding_constructor = embeddings[embedding]
            self._embedding = embedding_constructor(**embedding_settings[embedding])

        except KeyError:
            print(f"Embedding named {embedding} not present in the list of available embeddings: {list(embeddings.keys())}.")

        except TypeError:
            print(f"Embedding configuration not consistent.")


        try:
            database = list(database_settings.keys())[0]
            database_constructor = databases[database]
            self._database = database_constructor(**database_settings[database], embedding=self._embedding)
        
        except KeyError:
            print(f"Database named {database} not present in the list of available databases: {list(databases.keys())}.")

        except TypeError:
            print(f"Embedding configuration not consistent.")



settings = toml.load("settings.toml")
#de = DataEmbedding(settings)
embedding = OpenAIEmbeddings(api_key="")
chroma = Chroma(embedding_function=embedding, persist_directory="persist",collection_name="pippo")
loader = DataFrameLoader(DataFrame([{"text": "Sono io.", "id": 1}]), "text")
record_manager = SQLRecordManager("db/text", db_url="sqlite:///pippo.sql")
record_manager.create_schema()
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
data = loader.load()
documents = splitter.split_documents(data)
index(documents, record_manager, chroma, cleanup="incremental", source_id_key="id")