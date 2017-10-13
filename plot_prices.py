import pandas
import matplotlib.pyplot as plt

data = pandas.read_csv("coins.csv", index_col=0).T
data.plot(kind="line")

plt.show()
