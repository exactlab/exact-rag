from dataemb import DataEmbedding
import toml
from time import sleep

settings = toml.load("settings.toml")
de = DataEmbedding(settings)
de.load("pippo")
de.load("pluto")
de.load("paperino")
ans = de.chat("Is 'pippo' present in the data?")
print(ans)
sleep(1)
ans = de.chat("Is 'mio nonno in cariola' present in the data?")
print(ans)