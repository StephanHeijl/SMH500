from __future__ import division
import sqlite3
import time
import json
import datetime
import pandas
import pprint
import numpy as np
import math

pandas.options.display.width = None
conn = sqlite3.connect('coins.db')
cursor = conn.cursor()

cursor.execute("""
SELECT * FROM coins
""")

basic_coin_data = cursor.fetchall()

basic_coin_data_l = []

for row in basic_coin_data:
    basic_coin_data_l.append(
        list(row) + [datetime.date.fromtimestamp(row[0]).isoformat()]
    )

df_ci = pandas.DataFrame(basic_coin_data_l, columns=["ts", "coin", "USD", "EUR", "BTC", "day"])
df_ci.set_index(["coin", "day"], inplace=True)

cursor.execute("""
SELECT ts, coin, volume_to FROM volumes
""")

volume_data = cursor.fetchall()
volume_data_l = []

for row in volume_data:
    volume_data_l.append(
        list(row) + [datetime.date.fromtimestamp(row[0]).isoformat()]
    )

df_vol = pandas.DataFrame(volume_data_l, columns=["ts", "coin", "volume", "day"])
df_vol.set_index(["coin", "day"], inplace=True)
df_vol.drop("ts", inplace=True, axis=1)

full_data = df_ci.join(df_vol)
full_data.loc[:, "market_cap"] = full_data.loc[:, "BTC"].values * full_data.loc[:, "volume"]
full_data.dropna(how="any", axis=0, inplace=True)

caps = {}

for coin, group in full_data.groupby("coin"):
    caps[coin] = np.nanmean(group.loc[:, "market_cap"].values)

caps = sorted(caps.items(), key=lambda x: x[1])
caps_sum = sum(dict(caps).values())

caps = [(c, v / caps_sum) for c, v in caps]
pprint.pprint(caps)

with open("relative_market_caps.json", "w") as f:
    json.dump(caps, f)
