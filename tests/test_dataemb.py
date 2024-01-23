import pytest
import requests

from typing import Any, Callable
from exact_rag.dataemb import Caller, DataEmbedding
from exact_rag.config import Embeddings, Databases, EmbeddingType, DatabaseType
from tests.test_config import get_embedding_toml, get_database_toml

@pytest.fixture
def get_id_callable() -> Callable[..., Any]:
    def id(**kwargs: Any) -> dict[str, Any]:
        return kwargs
    return id

@pytest.fixture
def get_input_args() -> dict[str, Any]:
    d = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    return d

def test_Caller_no_args(get_id_callable, get_input_args):
    caller = Caller(get_id_callable)
    assert caller(**get_input_args) == get_input_args

def test_Caller_swap(get_id_callable, get_input_args):
    swap = {"A": "a", "D": "d", "EE": "e"}
    input = {"A": 1, "b": 2, "c": 3, "D": 4, "EE": 5}
    caller = Caller(get_id_callable, swap)
    assert caller(**input) == get_input_args

def test_Caller_accept_only(get_id_callable, get_input_args):
    accept_only = ["b", "c", "e"]
    expected = {"b": 2, "c": 3, "e": 5}
    caller = Caller(get_id_callable, None, accept_only)
    assert caller(**get_input_args) == expected

def test_Caller_full(get_id_callable):
    accept_only = ["B", "C", "EE"]
    swap = {"B": "b", "C": "c", "EE": "e"}
    input = {"a": 1, "B": 2, "C": 3, "d": 4, "EE": 5}
    expected = {"b": 2, "c": 3, "e": 5}
    caller = Caller(get_id_callable, swap, accept_only)
    assert caller(**input) == expected

def is_elastic_available(db_url: str) -> bool:
    try:
        ans = requests.get(f"{db_url}/_cluster/health?pretty", timeout=1.)
        if not ans.ok:
            return False
        if ans.json().get("status") not in ["green", "yellow"]:
            return False
    except:
        return False
    
    return True

def test_DataEmbedding_init(get_embedding_toml, get_database_toml):
    embedding = Embeddings(**get_embedding_toml)
    if embedding.type == EmbeddingType.openai:
        if not embedding.api_key or embedding.api_key == "":
            pytest.skip(reason="OpenAI token not found.")

    database = Databases(**get_database_toml)
    if database.type == DatabaseType.elastic:
        if not is_elastic_available(database.url):
            pytest.skip(reason="Elasticsearch database not present.")

        
    DataEmbedding(embedding, database)