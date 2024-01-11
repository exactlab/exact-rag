from exact_rag.dataemb import DataEmbedding
import toml

settings = toml.load("test_settings.toml")
de = DataEmbedding(settings)
de.load("pippo")
de.load("pluto")
de.load("paperino")
ans = de.chat("Is 'pippo' present in the data?")
print(ans)
ans = de.chat("Is 'topolino' present in the data?")
print(ans)
