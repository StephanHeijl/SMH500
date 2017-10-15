import requests
import json
import datetime
import time
import pandas
import numpy as np
from tqdm import tqdm

url = "https://min-api.cryptocompare.com/data/pricehistorical?fsym={coin}&tsyms=USD&ts={timestamp}&extraParams=SMH500"

coins = set(["BTC"])
with open("markets.json") as f:
    markets = json.load(f)["result"]
    for market in markets:
        market = market["Market"]
        if market["BaseCurrency"] == "BTC" and market["IsActive"]:
            coins.add(market["MarketCurrency"])

today = datetime.datetime.now()

timestamps = []
days = 52
day_no = 7
delay = 2
for i in range(days):
    d = today - datetime.timedelta(days=day_no * i)
    timestamps.append(int(time.mktime(d.timetuple())))

coins = list(coins)
#coins = ["BTC", "ETC"]
empty = np.empty((len(coins), days))
empty[:] = None
data = pandas.DataFrame(empty, index=coins, columns=timestamps)
timestamps.reverse()

for coin in tqdm(coins):
    for ts in tqdm(timestamps):
        #print url.format(coin=coin, timestamp=ts)
        done = False
        while not done:
            try:
                r = requests.get(url.format(coin=coin, timestamp=ts))
                data.loc[coin, ts] = r.json().values()[0]["USD"]
                done = True
            except:
                time.sleep(10)
        time.sleep(delay)

data.to_csv("coins.csv")
