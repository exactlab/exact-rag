import pytest
import toml

from exact_rag.config import Embeddings, Databases
from tests.conftest import embedding_tomls, database_tomls

def test_Embeddings(get_embedding_toml):
    Embeddings(**get_embedding_toml)

def test_Databases(get_database_toml):
    Databases(**get_database_toml)

omit_fields = [
    (Embeddings, embedding_tomls["ollama"], "embedding", "model"),
    (Databases, database_tomls["chroma"], "database", "persist_directory"),
    (Databases, database_tomls["elastic"], "database", "url"),
    (Databases, database_tomls["elastic"], "database", "distance_strategy"),
]
@pytest.mark.parametrize("dataclass, file, sub, omit", omit_fields)
def test_omit_fields(dataclass, file, sub, omit):
    settings = toml.load(f"{file}")
    settings_s = settings[sub]
    settings_s[omit] = None
    with pytest.raises(ValueError):
        dataclass(**settings_s)