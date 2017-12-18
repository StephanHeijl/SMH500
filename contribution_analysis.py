import sqlite3
import time
import json
import datetime
import pandas
import pprint
import numpy as np
import matplotlib
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from tqdm import tqdm
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pandas.options.display.width = None
conn = sqlite3.connect('coins.db')
cursor = conn.cursor()

def normalize(df):
    n_df = df - df.mean() + 1e-6
    n_df = n_df / n_df.std() + 1e-6
    return n_df

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
#df_vol = df_vol[df_vol.coin != "BTC"]
df_vol.set_index(["coin", "day"], inplace=True)
df_vol.drop("ts", inplace=True, axis=1)

full_data = df_ci.join(df_vol)
full_data.loc[:, "market_cap"] = full_data.loc[:, "USD"].values * full_data.loc[:, "volume"]
full_data.dropna(how="any", axis=0, inplace=True)

market_caps = full_data.loc[:, "market_cap"].reset_index()
total_volume_over_time = market_caps.groupby("day").sum()

total_volume_over_time.loc[:, "market_cap"] = normalize(total_volume_over_time.loc[:, "market_cap"])
distances = {}
coins_bar = tqdm(market_caps.coin.unique())
volumes = {}

for comp_coin in coins_bar:
    coins_bar.set_description(comp_coin)
    comp_coin_volume_over_time = market_caps.set_index("coin").loc[comp_coin].set_index("day")
    try:
        total_volume_over_time.loc[:, "market_cap_%s" % comp_coin] = normalize(
            comp_coin_volume_over_time.loc[:, "market_cap"]
        )
    except:
        continue

    comp_coin_v = np.nan_to_num(total_volume_over_time.loc[:, "market_cap_%s" % comp_coin].values)
    distance, path = fastdtw(
        np.nan_to_num(total_volume_over_time.loc[:, "market_cap"].values),
        np.nan_to_num(comp_coin_v),
        dist=euclidean
    )
    distances[comp_coin] = (distance, comp_coin_v[-1] - comp_coin_v[0])

# print(distances["BTC"], distances["ETH"], distances["DASH"])
distances = sorted(distances.items(), key=lambda x: x[1][1])
pprint.pprint(distances[:25])
pprint.pprint(distances[-25:])
# print(total_volume_over_time.columns)
total_volume_over_time.loc[:, [
    "market_cap",
    "market_cap_XDN",
    "market_cap_XEM",
    "market_cap_SNGLS",
    "market_cap_XRP",
    "market_cap_BNT"
]].plot(kind="line", figsize=(20, 10))
plt.savefig("market_caps.png")

with open("volume_contribution.json", "w") as f:
    json.dump(distances, f)
