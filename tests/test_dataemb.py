import pytest
import requests
import toml
import os

from typing import Any, Callable
from exact_rag.dataemb import Caller, DataEmbedding
from exact_rag.config import Embeddings, Databases, EmbeddingType, DatabaseType
from tests.conftest import embedding_tomls, database_tomls


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
        ans = requests.get(f"{db_url}/_cluster/health?pretty", timeout=1.0)
        if not ans.ok:
            return False
        if ans.json().get("status") not in ["green", "yellow"]:
            return False
    except Exception:
        return False

    return True


is_elasticserver_up = True


@pytest.mark.skipif(
    not is_elasticserver_up,
    reason="This test is active only if one has certancy that elasticserach server is up",
)
def test_is_elastic_available():
    settings = toml.load(database_tomls["elastic"])
    database = Databases(**settings["database"])
    is_elastic_available(database.url) is True
    is_elastic_available("http://localhost:22") is False


def get_elastic_indices(db_url: str) -> int:
    try:
        d = requests.get(f"{db_url}/*").json()
        return len(d)
    except Exception:
        return 0


def is_persistent_dir_not_empty(path: str) -> bool:
    return len(os.listdir(path)) != 0


def activate_elastic_destructive_wildcard(db_url: str) -> bool:
    header = {"Content-Type": "application/json"}
    request = {"transient": {"action.destructive_requires_name": False}}
    try:
        ans = requests.put(f"{db_url}/_cluster/settings", headers=header, json=request)
        return ans.ok
    except Exception:
        return False


def delete_elastic_indices(db_url: str) -> bool:
    try:
        ans = requests.delete(f"{db_url}/_all")
        return ans.ok
    except Exception:
        return False


def delete_persistent_dir_content(path: str) -> bool:
    try:
        for file in os.listdir(path):
            os.remove(file)
    except Exception:
        return False
    return True


def delete_duplicates_file(path: str) -> bool:
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
    except Exception:
        return False


@pytest.mark.skipif(
    not is_elasticserver_up,
    reason="This test is active only if one has certancy that elasticserach server is up",
)
def test_elastic_indices():
    e_settings = toml.load(embedding_tomls["ollama"])
    d_settings = toml.load(database_tomls["elastic"])
    embedding = Embeddings(**e_settings["embedding"])
    database = Databases(**d_settings["database"])

    if not is_elastic_available(database.url):
        pytest.skip(reason="Elastisearch database not available.")

    if get_elastic_indices(database.url) != 0:
        pytest.skip(reason="Elasticsearch database is not empty.")

    de = DataEmbedding(embedding, database)
    de.load("first")
    de.load("second")
    de.load("third")
    de.chat("Is there 'first' in my text collection?")
    assert activate_elastic_destructive_wildcard(database.url) is True
    assert get_elastic_indices(database.url) != 0
    assert delete_elastic_indices(database.url) is True
    assert get_elastic_indices(database.url) == 0
    assert delete_duplicates_file(database.sql_url) is True


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
