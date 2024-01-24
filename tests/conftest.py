import pytest
import toml


embedding_tomls = {
    "openai": "tests/test_embedding_openai.toml",
    "ollama": "tests/test_embedding_ollama.toml",
}


@pytest.fixture(params=list(embedding_tomls.values()))
def get_embedding_toml(request):
    settings = toml.load(f"{request.param}")
    return settings["embedding"]


database_tomls = {
    "chroma": "tests/test_database_chroma.toml",
    "elastic": "tests/test_database_elastic.toml",
}


@pytest.fixture(params=list(database_tomls.values()))
def get_database_toml(request):
    settings = toml.load(f"{request.param}")
    return settings["database"]
