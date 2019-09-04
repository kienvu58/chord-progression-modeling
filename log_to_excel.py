import tinydb
import pandas as pd

db = tinydb.TinyDB("logs/logs.json")
logs = db.all()
df = pd.DataFrame(logs)
df.to_excel("logs/logs.xlsx")