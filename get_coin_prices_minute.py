import requests
import json
import datetime
import time
import sqlite3
from tqdm import tqdm

url = "https://min-api.cryptocompare.com/data/histominute?fsym={coin}&tsym=USD&limit=1440&toTs={timestamp}&aggregate=1&e=CCCAGG"

coins = set(["BTC"])
with open("markets.json") as f:
    markets = json.load(f)["result"]
    for market in markets:
        market = market["Market"]
        if market["BaseCurrency"] == "BTC" and market["IsActive"]:
            coins.add(market["MarketCurrency"])

today = datetime.datetime.now()

timestamps = []
days = 4
day_no = 1
delay = 2
for i in range(days):
    d = today - datetime.timedelta(days=day_no * i)
    timestamps.append(int(time.mktime(d.timetuple())))

coins = list(coins)
timestamps.reverse()

conn = sqlite3.connect('coins_minute.db')
cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS
coins_minute(
    ts INTEGER,
    coin CHAR(10),
    high REAL,
    low REAL,
    open REAL,
    close REAL,
    volume_from REAL,
    volume_to REAL
)
""")

coins = coins[1:]
for coin in tqdm(coins):
    for ts in tqdm(timestamps):
        #print url.format(coin=coin, timestamp=ts)
        done = False
        while not done:
            try:
                r = requests.get(url.format(coin=coin, timestamp=ts))
                values = r.json().values()[0]
                for values in r.json()["Data"]:
                    values["coin"] = coin
                    cursor.execute("INSERT INTO coins_minute VALUES({time}, '{coin}', {high}, {low}, {open}, {close}, {volumefrom}, {volumeto})".format(**values))
                conn.commit()
                done = True
            except:
                time.sleep(10)
        time.sleep(delay)
