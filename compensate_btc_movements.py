import sqlite3
import time
import datetime
import pandas
import pprint
import numpy as np
from scipy.stats import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sbs
from tqdm import tqdm

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
for coin, group in df_ci.groupby("coin"):
    offset = np.roll(df_ci.loc[df_ci.loc[:, "coin"].values == coin, "USD"].values, 1)
    diff = df_ci.loc[df_ci.loc[:, "coin"].values == coin, "USD"].values - offset
    diff_p = diff / (df_ci.loc[df_ci.loc[:, "coin"].values == coin, "USD"].values + 1e-6)
    df_ci.loc[df_ci.loc[:, "coin"].values == coin, "diff"] = diff_p
    #df_ci.loc[df_ci.loc[:, "coin"].values == coin, "diff"] -= df_ci.loc[df_ci.loc[:, "coin"].values == coin, "diff"].min()
    #df_ci.loc[df_ci.loc[:, "coin"].values == coin, "diff"] /= df_ci.loc[df_ci.loc[:, "coin"].values == coin, "diff"].max()
    df_ci.loc[df_ci.loc[:, "coin"].values == coin, "diff_ma"] = pandas.rolling_mean(df_ci.loc[df_ci.loc[:, "coin"].values == coin, "diff"], 3)

df_ci.set_index(["coin"], inplace=True)

compare = "SIB"
BTC = df_ci.loc["IOP", "diff_ma"].values[30:-1]
COMP = df_ci.loc[compare, "diff_ma"].values[30:-1]

hm = True
if hm:
    coins = list(set(df_ci.index.values))
    distances = np.zeros((len(coins), len(coins)))
    for x, c1 in enumerate(tqdm(coins)):
        _c1 = df_ci.loc[c1, "diff_ma"].values[30:-1]
        for y, c2 in enumerate(tqdm(coins)):
            _c2 = df_ci.loc[c2, "diff_ma"].values[30:-1]
            if _c1.shape == _c2.shape:
                distances[x,y] = spearmanr(_c1, _c2)[0]
            else:
                distances[x,y] = 0

    print distances

    sbs.clustermap(distances,xticklabels=coins,yticklabels=coins, figsize=(32, 32))
    #sbs.heatmap(pandas.DataFrame(distances, index=coins, columns=coins))
    plt.savefig("coin_clusters.png", dpi=100)
    exit()
md = False
if md:

    mean_diffs = {}
    for coin, group in df_ci.groupby("coin"):
        _COIN = df_ci.loc[coin, "diff_ma"].values[10:-1]
        if _COIN.shape != BTC.shape:
            continue

        mean_diffs[coin] = spearmanr(BTC, _COIN)[0]

    pprint.pprint(sorted(mean_diffs.items(), key=lambda x: x[1]))

    exit()

plt.figure(figsize=(16, 9))
plt.plot(BTC, label="BTC")
plt.plot(COMP, label=compare)
#plt.plot(COMP - BTC, label="compensated %s" % compare)
plt.legend()
plt.savefig("compensate.png", dpi=100)
