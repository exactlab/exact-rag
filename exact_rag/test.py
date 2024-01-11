from dataemp import DataEmbedding
import toml

settings = toml.load("settings.toml")
de = DataEmbedding(settings)
de.load("pippo")
de.load("pluto")
de.load("paperino")
ans = de.chat("Is 'pippo' present in the data?")
print(ans)
ans = de.chat("Is 'mio nonno in cariola' present in the data?")
print(ans)
