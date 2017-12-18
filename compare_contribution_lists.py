import json
import pprint
import pandas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("volume_contribution.json") as f:
    volume_correspondence = json.load(f)
with open("relative_market_caps.json") as f:
    relative_market_caps = sorted(json.load(f))


overview = {}
for i, (coin, val) in enumerate(volume_correspondence):
    overview[coin] = [i]
for i, (coin, val) in enumerate(relative_market_caps):
    try:
        overview[coin].append(i)
    except:
        continue


overview = [(coin, vol, mar, (vol - mar) / 50 * 5 0, (vol + mar) / 10 * 10) for coin, (vol, mar) in overview.items()]
#pprint.pprint(overview)
# pprint.pprint(sorted(
#     overview,
#     key=lambda x: (x[-1], x[-2])
# ))

overview = pandas.DataFrame(overview, columns=["coin", "vol", "mar", "diff_vol_mar", "mean_vol_mar"])
overview.loc[:, ["vol", "mar"]].plot(x="vol", y="mar", kind="scatter", figsize=(20, 20))
plt.savefig("vol_mar_corr.png")
with pandas.option_context('display.max_rows', None, 'display.max_columns', 10):
    print overview.sort_values(["diff_vol_mar","mar"])
