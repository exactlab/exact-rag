from dataemb import DataEmbedding
import toml
import time

settings = toml.load("settings.toml")
de = DataEmbedding(settings)
#de.load("FUFFI is a ship")
ans = de.chat("What is FUFFI based on the context?")
print(ans)
# time.sleep(1)
# ans = de.chat("Is 'mio nonno in cariola' present in the data?")
# print(ans)
