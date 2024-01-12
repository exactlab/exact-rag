import toml
from pydantic import BaseModel

user_settings = toml.load("settings.toml")


class Embeddings(BaseModel):
    embedding_type: str = user_settings["embedding"].get("type")
    type: str | None = user_settings["embedding"].get("api_key")
    api_key: str | None = user_settings["embedding"].get("api_key")

    # chat: str | None = user_settings["embedding"]["chat"]["model_name"] | None
    # temperature: str | None = user_settings["embedding"]["chat"]["temperature"] | None
    # chain_type: str | None = user_settings["embedding"]["chain_type"] | None
    # search_type: str | None = user_settings["embedding"]["search"]["type"] | None
    # search_k: int = user_settings["embedding"]["search"]["k"] | 1
    # search_fetch_k: int = user_settings["embedding"]["search"]["fetch_k"] | 1


class Database(BaseModel):
    type: str


d = Database(type="pippo")
e = Embeddings()
print(e.model_dump())
