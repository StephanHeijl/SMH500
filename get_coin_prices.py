import requests
import json
import datetime
import time
import pandas
import numpy as np
import sqlite3
from tqdm import tqdm

url = "https://min-api.cryptocompare.com/data/pricehistorical?fsym={coin}&tsyms=USD,EUR,BTC&ts={timestamp}&extraParams=SMH500"

coins = set(["BTC"])
with open("markets.json") as f:
    markets = json.load(f)["result"]
    for market in markets:
        market = market["Market"]
        if market["BaseCurrency"] == "BTC" and market["IsActive"]:
            coins.add(market["MarketCurrency"])

today = datetime.datetime.now()

timestamps = []
days = 365
day_no = 1
delay = 2
for i in range(days):
    d = today - datetime.timedelta(days=day_no * i)
    timestamps.append(int(time.mktime(d.timetuple())))

coins = list(coins)
timestamps.reverse()

conn = sqlite3.connect('coins.db')
cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS
coins(
    ts INTEGER,
    coin CHAR(10),
    USD REAL,
    EUR REAL,
    BTC REAL
)
""")


for coin in tqdm(coins):
    for ts in tqdm(timestamps):
        #print url.format(coin=coin, timestamp=ts)
        done = False
        while not done:
            try:
                r = requests.get(url.format(coin=coin, timestamp=ts))
                values = r.json().values()[0]
                values["ts"] = int(ts)
                values["coin"] = coin
                cursor.execute("INSERT INTO coins VALUES({ts}, '{coin}', {USD}, {EUR}, {BTC})".format(**values))
                conn.commit()
                done = True
            except:
                sleep(10)
        time.sleep(delay)

data.to_csv("coins.csv")
