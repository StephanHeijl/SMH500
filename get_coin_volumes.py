import requests
import json
import datetime
import time
import sqlite3
from tqdm import tqdm

url = "https://min-api.cryptocompare.com/data/histoday?fsym={coin}&tsym=USD&limit=365&e=CCCAGG&extraParams=SMH500"
delay = 2

coins = set(["BTC"])
with open("markets.json") as f:
    markets = json.load(f)["result"]
    for market in markets:
        market = market["Market"]
        if market["BaseCurrency"] == "BTC" and market["IsActive"]:
            coins.add(market["MarketCurrency"])

today = datetime.datetime.now()

coins = list(coins)

conn = sqlite3.connect('coins.db')
cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS
volumes(
    ts INTEGER,
    coin CHAR(10),
    high REAL,
    low REAL,
    close REAL,
    open REAL,
    volume_from REAL,
    volume_to REAL
)
""")


for coin in tqdm(coins):

    done = False
    while not done:
        try:
            r = requests.get(url.format(coin=coin))
            values = r.json()["Data"]
            for v in values:
                v["coin"] = coin
                cursor.execute("INSERT INTO volumes VALUES({time}, '{coin}', {high}, {low}, {close}, {open}, {volumefrom}, {volumeto})".format(**v))
            conn.commit()
            done = True
        except Exception as e:
            print e
            time.sleep(10)
    time.sleep(delay)
