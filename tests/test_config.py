import pytest
import toml

from exact_rag.config import Embeddings, Databases


embedding_tomls = ["tests/test_embedding_openai.toml", "tests/test_embedding_ollama.toml"]


@pytest.fixture(params=embedding_tomls)
def get_embedding_toml(request):
    settings = toml.load(f"{request.param}")
    return settings["embedding"]


def test_Embeddings(get_embedding_toml):
    Embeddings(**get_embedding_toml)


database_tomls = ["tests/test_database_chroma.toml", "tests/test_database_elastic.toml"]


@pytest.fixture(params=database_tomls)
def get_database_toml(request):
    settings = toml.load(f"{request.param}")
    return settings["database"]


def test_Databases(get_database_toml):
    Databases(**get_database_toml)


omit_fields = [
    (Embeddings, embedding_tomls[1], "embedding", "model"),
    (Databases, database_tomls[0], "database", "persist_directory"),
    (Databases, database_tomls[1], "database", "url"),
    (Databases, database_tomls[1], "database", "distance_strategy"),
]


@pytest.mark.parametrize("dataclass, file, sub, omit", omit_fields)
def test_omit_fields(dataclass, file, sub, omit):
    settings = toml.load(f"{file}")
    settings_s = settings[sub]
    settings_s[omit] = None
    with pytest.raises(ValueError):
        dataclass(**settings_s)